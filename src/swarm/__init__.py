"""swarm: many agents collaborating to synthesise kernels at inference time."""

from swarm.generate import (
    KernelCache,
    LLMClient,
    TritonBackend,
    VerifyResult,
    WorkloadSignature,
    signature_of,
    synthesise_kernel,
    verify,
)
from swarm.council import CouncilBackend, CouncilConfig

__version__ = "0.0.0"

__all__ = [
    "CouncilBackend",
    "CouncilConfig",
    "KernelCache",
    "LLMClient",
    "TritonBackend",
    "VerifyResult",
    "WorkloadSignature",
    "signature_of",
    "synthesise_kernel",
    "verify",
]
