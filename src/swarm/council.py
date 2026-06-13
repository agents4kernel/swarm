"""swarm.council: best of N council synthesis layered over the generate primitives."""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import asdict, dataclass, field
from typing import Callable

import torch

from swarm.generate import (
    EXEMPLARS,
    SYSTEM_PROMPT,
    KernelCache,
    LLMClient,
    VerifyResult,
    WorkloadSignature,
    _compile_module,
    build_prompt,
    extract_code,
    verify,
)


STRATEGIES: dict[str, str] = {
    "coalesce": (
        "Maximize global memory coalescing. Consecutive lanes must touch consecutive "
        "addresses, give each program a contiguous tile, pad BLOCK to a warp multiple."
    ),
    "autotune": (
        "Wrap the entry in triton.autotune over a small lattice of BLOCK and num_warps "
        "keyed on the dynamic problem dims. Do not over enumerate the space."
    ),
    "fuse": (
        "Single launch, no intermediate tensors. Hold the reduction tree in registers or "
        "shared memory and fold the epilogue into the same kernel."
    ),
    "persistent": (
        "Persistent grid: one program per SM, grid stride over tiles, amortize launch and "
        "prologue across the wave."
    ),
    "vectorize": (
        "Widen the memory path. Issue the largest legal vector load and store per lane, "
        "mask only the ragged tail, drive toward peak DRAM bandwidth."
    ),
}

_EXEMPLAR_POOL: tuple[str, ...] = tuple(EXEMPLARS.keys())


@dataclass
class CouncilConfig:
    arms: int = 4
    max_rounds: int = 3
    explore_temperatures: tuple[float, ...] = (0.2, 0.45, 0.7, 0.95)
    exploit_temperature: float = 0.2
    rtol: float = 1e-2
    atol: float = 1e-2

    def arm_plan(self, round_index: int) -> list[tuple[str, float]]:
        names = list(STRATEGIES.keys())
        plan: list[tuple[str, float]] = []
        for k in range(self.arms):
            strategy = names[k % len(names)]
            if round_index == 0:
                temperature = self.explore_temperatures[k % len(self.explore_temperatures)]
            else:
                temperature = self.exploit_temperature
            plan.append((strategy, temperature))
        return plan


@dataclass
class Candidate:
    arm: int
    round_index: int
    strategy: str
    temperature: float
    source: str
    result: VerifyResult
    run: Callable[..., torch.Tensor] | None = None

    @property
    def ok(self) -> bool:
        return self.result.ok and self.run is not None

    def objectives(self) -> tuple[float, float, float]:
        return (
            self.result.speedup_vs_reference,
            -self.result.max_abs_diff,
            -float(len(self.source)),
        )


@dataclass
class CouncilTranscript:
    rounds: list[list[dict]] = field(default_factory=list)

    def record(self, candidates: list[Candidate]) -> None:
        self.rounds.append(
            [
                {
                    "arm": c.arm,
                    "strategy": c.strategy,
                    "temperature": c.temperature,
                    "ok": c.ok,
                    "speedup": c.result.speedup_vs_reference,
                    "max_abs_diff": c.result.max_abs_diff,
                    "bytes": len(c.source),
                    "error": (c.result.error or "")[:200],
                }
                for c in candidates
            ]
        )


def _dominates(a: Candidate, b: Candidate) -> bool:
    oa, ob = a.objectives(), b.objectives()
    return all(x >= y for x, y in zip(oa, ob)) and any(x > y for x, y in zip(oa, ob))


def _pareto_front(candidates: list[Candidate]) -> list[Candidate]:
    return [c for c in candidates if not any(_dominates(o, c) for o in candidates if o is not c)]


def _elect(correct: list[Candidate]) -> Candidate:
    return max(_pareto_front(correct), key=lambda c: c.objectives())


def _failure_rank(c: Candidate) -> tuple[bool, bool, float]:
    res = c.result
    produced_output = res.max_abs_diff == res.max_abs_diff
    compiled = "compile:" not in (res.error or "")
    closeness = res.max_abs_diff if produced_output else float("inf")
    return (not produced_output, not compiled, closeness)


class CouncilBackend:
    name = "triton-council"

    def __init__(
        self,
        llm: LLMClient | None = None,
        cache: KernelCache | None = None,
        config: CouncilConfig | None = None,
    ) -> None:
        self.llm = llm or LLMClient()
        self.cache = cache or KernelCache()
        self.config = config or CouncilConfig()

    def synthesise(
        self,
        sig: WorkloadSignature,
        reference: Callable[..., torch.Tensor],
        reference_source: str,
        example_inputs: tuple[torch.Tensor, ...],
        max_attempts: int = 4,
    ) -> tuple[Callable[..., torch.Tensor], VerifyResult, str]:
        cached = self.cache.get(sig)
        if cached:
            ns = _compile_module(cached["source"])
            return ns["run"], cached["verify"], cached["source"]

        cfg = self.config
        transcript = CouncilTranscript()
        seeds: list[Candidate] = []
        best_failure: Candidate | None = None

        for round_index in range(cfg.max_rounds):
            prompts = self._round_prompts(sig, reference_source, seeds, cfg.arm_plan(round_index))
            replies = self._fan_out(prompts)
            candidates = self._adjudicate(prompts, replies, round_index, reference, example_inputs)
            transcript.record(candidates)

            correct = [c for c in candidates if c.ok]
            if correct:
                champion = _elect(correct)
                self.cache.put(
                    sig,
                    {
                        "source": champion.source,
                        "verify": champion.result,
                        "sig": asdict(sig),
                        "council": asdict(transcript),
                    },
                )
                assert champion.run is not None
                return champion.run, champion.result, champion.source

            ranked = sorted(candidates, key=_failure_rank)
            if ranked and (best_failure is None or _failure_rank(ranked[0]) < _failure_rank(best_failure)):
                best_failure = ranked[0]
            seeds = ranked[: cfg.arms]

        reason = best_failure.result.error if best_failure else "no candidates produced"
        raise RuntimeError(f"council exhausted {cfg.max_rounds} rounds, best: {reason}")

    def _round_prompts(
        self,
        sig: WorkloadSignature,
        reference_source: str,
        seeds: list[Candidate],
        plan: list[tuple[str, float]],
    ) -> list[tuple[int, str, float, str]]:
        prompts: list[tuple[int, str, float, str]] = []
        for arm, (strategy, temperature) in enumerate(plan):
            seed = seeds[arm % len(seeds)] if seeds else None
            prompt = build_prompt(
                sig=sig,
                reference_source=reference_source,
                exemplar_names=self._exemplars_for(arm),
                last_attempt=seed.source if seed else None,
                last_error=seed.result.error if seed else None,
            )
            prompt = f"{prompt}\n\n## Strategy directive\n{STRATEGIES[strategy]}"
            prompts.append((arm, strategy, temperature, prompt))
        return prompts

    def _adjudicate(
        self,
        prompts: list[tuple[int, str, float, str]],
        replies: list[object],
        round_index: int,
        reference: Callable[..., torch.Tensor],
        example_inputs: tuple[torch.Tensor, ...],
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for (arm, strategy, temperature, _), reply in zip(prompts, replies):
            if isinstance(reply, Exception):
                result = VerifyResult(ok=False, error=f"llm: {reply}")
                candidates.append(Candidate(arm, round_index, strategy, temperature, "", result))
                continue
            try:
                source = extract_code(str(reply))
            except ValueError as exc:
                candidates.append(
                    Candidate(arm, round_index, strategy, temperature, str(reply), VerifyResult(ok=False, error=str(exc)))
                )
                continue
            result, run = verify(source, reference, example_inputs, self.config.rtol, self.config.atol)
            candidates.append(Candidate(arm, round_index, strategy, temperature, source, result, run))
        return candidates

    def _exemplars_for(self, arm: int) -> tuple[str, ...]:
        if len(_EXEMPLAR_POOL) <= 1:
            return _EXEMPLAR_POOL
        pivot = arm % len(_EXEMPLAR_POOL)
        return _EXEMPLAR_POOL[pivot:] + _EXEMPLAR_POOL[:pivot]

    def _fan_out(self, prompts: list[tuple[int, str, float, str]]) -> list[object]:
        out: list[object] = [None] * len(prompts)
        with cf.ThreadPoolExecutor(max_workers=max(1, len(prompts))) as pool:
            futures = {
                pool.submit(self.llm.complete, SYSTEM_PROMPT, prompt, temperature=temperature): index
                for index, (_, _, temperature, prompt) in enumerate(prompts)
            }
            for future in cf.as_completed(futures):
                index = futures[future]
                try:
                    out[index] = future.result()
                except Exception as exc:  # noqa: BLE001
                    out[index] = exc
        return out
