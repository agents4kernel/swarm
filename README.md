<div align="center">
  <img src="assets/swarm.svg" alt="swarm" width="420"/>

  <p><strong>The runtime that writes the kernel your inference workload deserves, one request at a time.</strong></p>

  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-0F172A?style=flat-square" alt="License"/></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square" alt="Python 3.11+"/></a>
    <a href="https://pypi.org/project/swarm-kernels/"><img src="https://img.shields.io/pypi/v/swarm-kernels?style=flat-square&color=0F172A" alt="PyPI"/></a>
    <a href="https://discord.gg/swarm"><img src="https://img.shields.io/badge/discord-join-5865F2?style=flat-square" alt="Discord"/></a>
  </p>

  <p>
    <a href="#quickstart">Quickstart</a> ·
    <a href="https://discord.gg/swarm">Discord</a>
  </p>
</div>

---

## What is Swarm?

A multi agent kernel synthesis runtime for AI inference. Workload aware. Cross accelerator. Verified per call.

## Quickstart

### Prerequisites

Python 3.11 or newer. An accelerator and its toolchain (CUDA, ROCm, or Apple Silicon). An LLM API key.

```bash
pip install swarm-kernels[nvidia]
export ANTHROPIC_API_KEY=sk-ant-...
```

### Synthesise a kernel

```bash
python -m swarm.generate --op softmax --shape 8192 8192 --dtype float16
```

```json
{
  "op": "softmax",
  "shape": [8192, 8192],
  "dtype": "float16",
  "ok": true,
  "max_abs_diff": 0.0009765625,
  "speedup_vs_reference": 1.42,
  "kernel_ms": 0.183,
  "reference_ms": 0.260
}
```

### Use it from Python

```python
import torch
from swarm import synthesise_kernel

def my_softmax(x):
    return torch.softmax(x, dim=-1)

x = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
run, result = synthesise_kernel("softmax", my_softmax, (x,))

y = run(x)
print(f"{result.speedup_vs_reference:.2f}x vs torch reference")
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact

- [GitHub Issues](https://github.com/swarm-ai/swarm/issues)
- [Discord](https://discord.gg/swarm)

## License

Apache 2.0. See [LICENSE](LICENSE).
