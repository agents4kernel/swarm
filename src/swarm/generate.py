"""
swarm.generate

Inference time kernel synthesis loop. One file, end to end:

    reference torch fn  ->  signature hash  ->  cache hit?  ->  done
                                              ->  cache miss
                                                    ->  LLM prompt with exemplars
                                                    ->  compile
                                                    ->  correctness gate vs torch
                                                    ->  perf gate vs torch reference
                                                    ->  retry with error feedback
                                                    ->  persist to cache  ->  done

Patterns lifted from:
  KernelBench, Princeton, https://github.com/ScalingIntelligence/KernelBench
      eval harness shape, correctness via torch.allclose, three difficulty levels
  KernelLLM, Meta,        https://github.com/meta-pytorch/KernelLLM
      Triton-first prompt structure, single-turn synthesis target
  AI CUDA Engineer, Sakana, https://pub.sakana.ai/static/paper.pdf
      iterative refine with execution feedback, correctness reward shaping
      (we add hard correctness gating after their reward-hacking incident)

This is a Triton-on-NVIDIA reference path. CUDA, ROCm, Metal, and ASIC
backends would each implement the same KernelBackend protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import tempfile
import textwrap
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

import torch
import torch.nn.functional as F

# Anthropic is the default. Swap to openai / litellm trivially.
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from swarm.council import CouncilConfig


# =============================================================================
# 1. Workload signature.  The cache key.
# =============================================================================

@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    device: str

    @classmethod
    def of(cls, t: torch.Tensor) -> "TensorSpec":
        return cls(tuple(t.shape), str(t.dtype).removeprefix("torch."), t.device.type)


@dataclass(frozen=True)
class WorkloadSignature:
    op_name: str
    inputs: tuple[TensorSpec, ...]
    target_compute_capability: str
    target_dtype_class: str
    kwargs_hash: str

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def signature_of(
    op_name: str,
    example_inputs: tuple[torch.Tensor, ...],
    kwargs: dict[str, Any] | None = None,
) -> WorkloadSignature:
    kwargs = kwargs or {}
    if torch.cuda.is_available():
        cc_major, cc_minor = torch.cuda.get_device_capability()
        cc = f"sm_{cc_major}{cc_minor}"
    else:
        cc = "cpu"
    dtype_class = str(example_inputs[0].dtype).removeprefix("torch.")
    kwargs_blob = json.dumps(kwargs, sort_keys=True, default=str)
    kwargs_h = hashlib.sha256(kwargs_blob.encode()).hexdigest()[:8]
    return WorkloadSignature(
        op_name=op_name,
        inputs=tuple(TensorSpec.of(t) for t in example_inputs),
        target_compute_capability=cc,
        target_dtype_class=dtype_class,
        kwargs_hash=kwargs_h,
    )


# =============================================================================
# 2. Kernel cache.  Signature hash -> persisted compiled artifact.
# =============================================================================

class KernelCache:
    def __init__(self, root: Path | str = "~/.cache/swarm") -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, sig: WorkloadSignature) -> Path:
        return self.root / f"{sig.op_name}-{sig.hash()}.kern"

    def get(self, sig: WorkloadSignature) -> dict[str, Any] | None:
        p = self._key(sig)
        if not p.exists():
            return None
        with p.open("rb") as fh:
            return pickle.load(fh)

    def put(self, sig: WorkloadSignature, artifact: dict[str, Any]) -> None:
        with self._key(sig).open("wb") as fh:
            pickle.dump(artifact, fh)


# =============================================================================
# 3. Exemplar bank.  Hand-written reference Triton kernels we paste into
# the prompt as few-shots.  This is where domain knowledge lives.
# =============================================================================

EXEMPLARS: dict[str, str] = {
    "elementwise_add": textwrap.dedent('''
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        def run(x, y):
            assert x.is_cuda and y.is_cuda and x.shape == y.shape
            out = torch.empty_like(x)
            n = x.numel()
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
            add_kernel[grid](x, y, out, n, BLOCK=1024)
            return out
    ''').strip(),
    "softmax_row": textwrap.dedent('''
        import triton
        import triton.language as tl

        @triton.jit
        def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                           n_cols, BLOCK_N: tl.constexpr):
            row_idx = tl.program_id(0)
            row_start = in_ptr + row_idx * in_row_stride
            offs = tl.arange(0, BLOCK_N)
            mask = offs < n_cols
            x = tl.load(row_start + offs, mask=mask, other=-float("inf"))
            x_max = tl.max(x, axis=0)
            x = x - x_max
            num = tl.exp(x)
            den = tl.sum(num, axis=0)
            y = num / den
            out_row = out_ptr + row_idx * out_row_stride
            tl.store(out_row + offs, y, mask=mask)

        def run(x):
            assert x.is_cuda and x.ndim == 2
            n_rows, n_cols = x.shape
            out = torch.empty_like(x)
            BLOCK_N = triton.next_power_of_2(n_cols)
            softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0),
                                      n_cols, BLOCK_N=BLOCK_N,
                                      num_warps=4)
            return out
    ''').strip(),
}


# =============================================================================
# 4. Prompt construction.  System + user messages.  Few-shots inline.
# =============================================================================

SYSTEM_PROMPT = """You are an expert GPU kernel author. You write Triton kernels that match the semantics of a PyTorch reference implementation and run as fast as the hardware allows.

Constraints, no exceptions:
  1. Output one Python source file as a fenced code block. No prose outside the fence.
  2. The file must define `run(*args, **kwargs)` that takes the same tensors as the reference and returns the same shape and dtype.
  3. Imports are limited to: `torch`, `triton`, `triton.language as tl`, and `math`.
  4. No external state. No global tensors. No print, no logging, no os, no subprocess.
  5. The kernel must be numerically equivalent to the reference at the given dtype within `rtol=1e-2, atol=1e-2`. Lower tolerance is better, do not abuse it.
  6. Prefer power of two BLOCK sizes. Pick `num_warps` based on the workload. Use `triton.autotune` if and only if the problem size genuinely benefits.

You are graded on: correctness first, speedup over the reference second, and code size third. Aim for code that an experienced Triton author would write."""


def build_prompt(
    sig: WorkloadSignature,
    reference_source: str,
    exemplar_names: tuple[str, ...] = ("elementwise_add", "softmax_row"),
    last_attempt: str | None = None,
    last_error: str | None = None,
) -> str:
    exemplar_block = "\n\n".join(
        f"### Exemplar: `{name}`\n```python\n{EXEMPLARS[name]}\n```"
        for name in exemplar_names
        if name in EXEMPLARS
    )
    inputs_block = "\n".join(
        f"  arg{i}: shape={spec.shape} dtype={spec.dtype} device={spec.device}"
        for i, spec in enumerate(sig.inputs)
    )
    retry_block = ""
    if last_attempt and last_error:
        retry_block = textwrap.dedent(f"""
            ### Previous attempt failed
            Your last kernel produced this error or correctness gap:
            ```
            {last_error.strip()[:1500]}
            ```
            That code was:
            ```python
            {last_attempt.strip()[:3000]}
            ```
            Fix the issue. Do not repeat the same mistake. Output a new full file.
            """).strip()

    return textwrap.dedent(f"""
        # Task: synthesise a Triton kernel for `{sig.op_name}`

        ## Target
          device compute capability: {sig.target_compute_capability}
          dtype class:               {sig.target_dtype_class}

        ## Inputs
        {inputs_block}

        ## Reference (PyTorch)
        ```python
        {reference_source.strip()}
        ```

        ## Exemplars to learn from
        {exemplar_block}

        {retry_block}

        ## Your output
        Produce one Python file as described in the system prompt.
        """).strip()


# =============================================================================
# 5. LLM call.  Anthropic.  Trivial to swap.
# =============================================================================

class LLMClient:
    def __init__(
        self,
        model: str = "claude-opus-4-8",
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> None:
        if Anthropic is None:
            raise RuntimeError(
                "Install with: pip install anthropic, or replace LLMClient."
            )
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system: str, user: str, temperature: float | None = None) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> str:
    m = _CODE_FENCE.search(reply)
    if not m:
        raise ValueError("model returned no fenced python block")
    return m.group(1).strip()


# =============================================================================
# 6. Verifier.  Correctness gate first, then perf.
# =============================================================================

@dataclass
class VerifyResult:
    ok: bool
    error: str | None = None
    max_abs_diff: float = float("nan")
    speedup_vs_reference: float = float("nan")
    kernel_ms: float = float("nan")
    reference_ms: float = float("nan")


def _compile_module(source: str) -> dict[str, Any]:
    """Load the generated source in a fresh namespace."""
    ns: dict[str, Any] = {}
    code = compile(source, "<generated_kernel>", "exec")
    exec(code, ns)
    if "run" not in ns or not callable(ns["run"]):
        raise RuntimeError("generated file does not define a callable `run`")
    return ns


def _cuda_time_ms(fn: Callable[[], Any], iters: int = 50, warmup: int = 10) -> float:
    if not torch.cuda.is_available():
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) * 1000.0 / iters
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = getattr(props, "L2_cache_size", 0) or (64 << 20)
    flush = torch.empty(int(l2_bytes), dtype=torch.int8, device="cuda")
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    samples = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return samples[len(samples) // 2]


def verify(
    source: str,
    reference: Callable[..., torch.Tensor],
    example_inputs: tuple[torch.Tensor, ...],
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> tuple[VerifyResult, Callable[..., torch.Tensor] | None]:
    try:
        ns = _compile_module(source)
        run = ns["run"]
    except Exception as e:
        return VerifyResult(ok=False, error=f"compile: {e}\n{traceback.format_exc()}"), None

    try:
        with torch.inference_mode():
            ref_out = reference(*example_inputs)
            kern_out = run(*example_inputs)
    except Exception as e:
        return VerifyResult(ok=False, error=f"runtime: {e}\n{traceback.format_exc()}"), None

    if ref_out.shape != kern_out.shape:
        return VerifyResult(ok=False, error=f"shape mismatch: ref {ref_out.shape} vs kern {kern_out.shape}"), None
    if ref_out.dtype != kern_out.dtype:
        return VerifyResult(ok=False, error=f"dtype mismatch: ref {ref_out.dtype} vs kern {kern_out.dtype}"), None

    diff = (ref_out.float() - kern_out.float()).abs()
    max_abs = diff.max().item()
    close = torch.allclose(ref_out, kern_out, rtol=rtol, atol=atol)
    if not close:
        return VerifyResult(ok=False, error=f"correctness fail: max_abs={max_abs:.6f}", max_abs_diff=max_abs), None

    ref_ms = _cuda_time_ms(lambda: reference(*example_inputs))
    ker_ms = _cuda_time_ms(lambda: run(*example_inputs))
    speed = ref_ms / max(ker_ms, 1e-9)
    return VerifyResult(
        ok=True,
        max_abs_diff=max_abs,
        speedup_vs_reference=speed,
        kernel_ms=ker_ms,
        reference_ms=ref_ms,
    ), run


# =============================================================================
# 7. Orchestrator.  The synthesis loop.
# =============================================================================

class KernelBackend(Protocol):
    """Backend protocol. Implemented for Triton, CUDA, ROCm, Metal, ASIC targets."""
    name: str

    def synthesise(
        self,
        sig: WorkloadSignature,
        reference: Callable[..., torch.Tensor],
        reference_source: str,
        example_inputs: tuple[torch.Tensor, ...],
        max_attempts: int = 4,
    ) -> tuple[Callable[..., torch.Tensor], VerifyResult, str]: ...


class TritonBackend:
    name = "triton"

    def __init__(
        self,
        llm: LLMClient | None = None,
        cache: KernelCache | None = None,
        council: "CouncilConfig | None" = None,
    ) -> None:
        self.llm = llm or LLMClient()
        self.cache = cache or KernelCache()
        self.council = council

    def synthesise(
        self,
        sig: WorkloadSignature,
        reference: Callable[..., torch.Tensor],
        reference_source: str,
        example_inputs: tuple[torch.Tensor, ...],
        max_attempts: int = 4,
    ) -> tuple[Callable[..., torch.Tensor], VerifyResult, str]:
        if self.council is not None:
            from swarm.council import CouncilBackend

            return CouncilBackend(self.llm, self.cache, self.council).synthesise(
                sig, reference, reference_source, example_inputs, max_attempts
            )
        cached = self.cache.get(sig)
        if cached:
            ns = _compile_module(cached["source"])
            return ns["run"], cached["verify"], cached["source"]

        last_src: str | None = None
        last_err: str | None = None
        last_res = VerifyResult(ok=False, error="no attempts made")

        for attempt in range(1, max_attempts + 1):
            user = build_prompt(
                sig=sig,
                reference_source=reference_source,
                last_attempt=last_src,
                last_error=last_err,
            )
            reply = self.llm.complete(SYSTEM_PROMPT, user)
            try:
                src = extract_code(reply)
            except ValueError as e:
                last_err = str(e)
                last_src = reply
                continue

            res, run = verify(src, reference, example_inputs)
            last_res = res
            if res.ok and run is not None:
                artifact = {"source": src, "verify": res, "sig": asdict(sig)}
                self.cache.put(sig, artifact)
                return run, res, src

            last_src = src
            last_err = res.error or "unknown failure"

        raise RuntimeError(f"synthesis failed after {max_attempts} attempts: {last_err}")


# =============================================================================
# 8. Top-level entry point.  This is what an inference server calls.
# =============================================================================

def synthesise_kernel(
    op_name: str,
    reference: Callable[..., torch.Tensor],
    example_inputs: tuple[torch.Tensor, ...],
    *,
    backend: KernelBackend | None = None,
    reference_source: str | None = None,
    max_attempts: int = 4,
) -> tuple[Callable[..., torch.Tensor], VerifyResult]:
    """Synthesise a kernel for `op_name` matching `reference` on `example_inputs`.

    Returns a tuple of (callable, verify result). Call the callable with the
    same arg layout as the reference function. Result is cached on disk by
    workload signature so subsequent calls return instantly.
    """
    backend = backend or TritonBackend()
    if reference_source is None:
        try:
            import inspect
            reference_source = inspect.getsource(reference)
        except OSError:
            reference_source = f"# reference source not available for {reference.__qualname__}"

    sig = signature_of(op_name, example_inputs)
    run, res, _ = backend.synthesise(
        sig=sig,
        reference=reference,
        reference_source=reference_source,
        example_inputs=example_inputs,
        max_attempts=max_attempts,
    )
    return run, res


# =============================================================================
# 9. CLI demo.  python -m swarm.generate --op softmax --shape 8192 8192
# =============================================================================

def _demo_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def _demo_layernorm(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, normalized_shape=(x.shape[-1],))


_DEMOS: dict[str, Callable[..., torch.Tensor]] = {
    "softmax": _demo_softmax,
    "layernorm": _demo_layernorm,
}


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="swarm kernel synthesiser")
    p.add_argument("--op", default="softmax", choices=list(_DEMOS.keys()))
    p.add_argument("--shape", type=int, nargs="+", default=[4096, 4096])
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attempts", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    x = torch.randn(*args.shape, dtype=dtype, device=args.device)
    ref = _DEMOS[args.op]

    run, res = synthesise_kernel(
        op_name=args.op,
        reference=ref,
        example_inputs=(x,),
        max_attempts=args.attempts,
    )
    print(json.dumps({
        "op": args.op,
        "shape": list(args.shape),
        "dtype": args.dtype,
        "ok": res.ok,
        "max_abs_diff": res.max_abs_diff,
        "speedup_vs_reference": res.speedup_vs_reference,
        "kernel_ms": res.kernel_ms,
        "reference_ms": res.reference_ms,
    }, indent=2))


if __name__ == "__main__":
    main()
