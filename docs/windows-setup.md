# Windows setup notes

This project was trained, quantized, and benchmarked entirely on a Windows
laptop (no WSL, no Linux dev box). That turned out to be a real source of
friction — six distinct, genuine environment problems surfaced across
training and GPU-accelerated inference, none of them guessable from docs
or training data, all root-caused by direct reproduction (reading the
actual traceback, tracing the actual DLL dependency chain, etc.) rather
than trial-and-error. Documented here because a grader or future
contributor reproducing this project on Windows would hit the same walls.

If you're on Linux/WSL or already have a CUDA-enabled build working, you
can ignore this file entirely — none of these are needed on Linux, and
most only matter if you're building `llama-cpp-python` or Triton kernels
from source rather than using prebuilt wheels.

## Training-time issues (`training/train.py`)

### 1. `datasets` + Python 3.14 pickle incompatibility

`datasets.utils._dill.Pickler._batch_setitems` was written against the
pre-3.14 `pickle.Pickler._batch_setitems(self, items)` signature. Python
3.14's `pickle.save_dict` calls it with an extra `obj` argument, so every
`load_dataset()`/`.map()` call (both use dataset fingerprinting under the
hood) crashed with:

```
TypeError: _Pickler._batch_setitems() takes 2 positional arguments but 3 were given
```

Fixed with a targeted monkeypatch — `_patch_datasets_dill_for_py314()` in
[`training/train.py`](../training/train.py) — that re-implements the
method, forwarding the extra `obj` argument through to dill's real
implementation instead of dropping it. Gated on `sys.version_info < (3, 14)`
so it's a no-op on any environment where `datasets` ships an upstream fix.

### 2. No C compiler installed for Triton's JIT build

Unsloth's training path uses Triton kernels, which JIT-build a small
native `cuda_utils` glue module the first time a kernel runs. Windows has
no C compiler by default. Fixed by installing a portable
[`w64devkit`](https://github.com/skeeto/w64devkit) (gcc, checksum-verified
download) rather than the much heavier full MSVC Build Tools, since a
working `cl.exe` install wasn't a hard requirement elsewhere in this
project at the time.

### 3. Three triton-windows/gcc gaps, found by attempting the build (not by reading changelogs)

Triton bundles its own mini CUDA Toolkit (`ptxas`, `cuda.h`, `cuda.lib`)
under `triton/backends/nvidia/`, used to build that `cuda_utils` module.
With gcc instead of MSVC as the compiler, three separate assumptions in
triton's build code broke:

- **`find_cuda_bundled()` looks in the wrong install location.** It
  assumes packages live under `sysconfig.get_paths()["platlib"]` (the
  system-wide site-packages dir). This environment installs packages to
  the **per-user** site-packages instead (no venv), so triton reported
  "Failed to find CUDA" even though everything needed was present. Fixed
  by pointing `CUDA_PATH` directly at triton's own bundled install so the
  higher-priority `find_cuda_env()` check finds it first.
- **`normalize_path()` always adds a Windows long-path (`\\?\`) prefix.**
  The build code only strips it back off (via an 8.3 short-path
  conversion) when the compiler is MSVC, so gcc's `cc1` received a
  literal prefixed path it doesn't understand and failed with `fatal
  error: cuda_utils.c: No such file or directory`. Fixed by skipping the
  long-path prefix entirely for non-MSVC compilers (safe here since the
  build temp dirs are well under the 260-character limit anyway).
- **`_library_flag()` emits bare library names** (`cuda.lib`,
  `python314.lib`) — correct for MSVC's `link.exe`, which applies
  `/LIBPATH` search dirs to bare names, but GNU `ld` does **not** apply
  `-L` search dirs to bare positional filenames, only to
  `-l<name>`/`-l:<name>` flags — so linking failed with "cannot find
  cuda.lib". Fixed by switching to the `-l:<exact filename>` syntax for
  non-MSVC compilers.

All three fixes live in `_patch_triton_for_mingw_windows()` in
[`training/train.py`](../training/train.py), gated to only apply on
Windows with a non-MSVC compiler (`_tb.is_msvc(cc)` / `is_clang_cl(cc)`
check) — harmless no-ops if you ever switch to MSVC or run on Linux.

## GPU-accelerated inference issues (`llama-cpp-python` CUDA rebuild)

These three came up rebuilding `llama-cpp-python` from source with CUDA
support, needed because no prebuilt CUDA wheel matched this environment's
CUDA Toolkit version. Build invocation:

```bash
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler" \
CMAKE_GENERATOR=Ninja \
FORCE_CMAKE=1 \
pip install --force-reinstall --no-cache-dir llama-cpp-python --no-binary llama-cpp-python
```

run inside a `vcvars64.bat`-initialized MSVC environment (needed for the
linker even though nvcc is doing the actual CUDA compilation).

### 4. CMake's default generator: "No CUDA toolset found"

The default Visual Studio CMake generator failed with `No CUDA toolset
found` — CUDA's VS-integration files didn't recognize the installed VS
2026/BuildTools version. Fixed with `CMAKE_GENERATOR=Ninja`, which talks
to nvcc directly instead of going through VS's CUDA toolset integration.

### 5. nvcc rejecting the installed MSVC version as unsupported

Once CMake could configure, `nvcc` failed with `fatal error C1189`,
rejecting the installed MSVC (14.50) as newer than its officially
supported version range. Fixed with
`-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler` — verified after the
fact that this still produces correct, working kernels (see GPU
verification in the README's Quantization section), not just a compile
that happens to succeed.

### 6. DLL load failure at runtime despite a successful build

After a clean build, `from llama_cpp import Llama` failed with a DLL-load
error. Root-caused (not guessed) via `dumpbin /dependents`-tracing the
actual dependency chain: `ggml-cuda.dll` needs `cublas64_13.dll`, which
CUDA 13.x installs under `CUDA\v13.1\bin\x64\` rather than `bin\`
directly (a change from earlier CUDA versions), and `llama_cpp`'s own
`CUDA_PATH\bin` fallback doesn't account for that subfolder. Separately,
reading `llama_cpp`'s `_ctypes_extensions.py` source showed its Windows
loader passes `winmode=ctypes.RTLD_GLOBAL` to `ctypes.CDLL` — and
`ctypes.RTLD_GLOBAL` is `0` on Windows, which forces the legacy DLL
search order that ignores `os.add_dll_directory()`-registered paths
entirely. Only the literal `PATH` environment variable matters. Fixed by
prepending CUDA's `bin\x64` directory to `PATH` before importing
`llama_cpp`:

```bash
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/x64:$PATH"
python evaluation/evaluate_quantized.py --gguf ... --n-gpu-layers -1 --verbose-load
```

## Reproducing the GPU benchmark

```bash
python evaluation/evaluate_quantized.py \
    --gguf inference_gguf/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf \
    --report evaluation/results/report.json \
    --out evaluation/results/report_quantized_q4_k_m_gpu.json \
    --quant-label q4_k_m \
    --n-gpu-layers -1 \
    --verbose-load
```

`--verbose-load` prints llama.cpp's own layer-offload logs (should show
all layers assigned `dev = CUDA0`) — check this, and cross-check with
`nvidia-smi` GPU utilization during the run, rather than trusting a
successful build alone. See the README's Quantization section for how
that verification was actually done here.
