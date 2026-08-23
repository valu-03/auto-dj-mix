"""Making onnxruntime actually find CUDA on Windows.

The NVIDIA pip wheels (`nvidia-cublas-cu12` and friends) drop their DLLs into
`site-packages/nvidia/<lib>/bin`, which Windows does not search by default.

`os.add_dll_directory()` is the documented fix and it does NOT work here:
onnxruntime loads `onnxruntime_providers_cuda.dll` from its own native code,
which does not consult the directories Python added. Measured -- with
add_dll_directory alone, CUDA still failed on a missing `cublasLt64_12.dll`
that was sitting in a directory we had just added.

Prepending those directories to PATH *before* onnxruntime is imported does work,
because the native loader does search PATH.

Why this matters more than a normal setup detail: when CUDA fails to load,
onnxruntime does not raise. It silently returns a CPU session. Every separation
would run ~15x slower with no error -- a job that takes four hours instead of
fifteen minutes and gives no reason why. So `cuda_status()` exists to prove the
provider is genuinely active rather than merely listed.
"""

import glob
import os
import site

_enabled = False


def nvidia_dll_dirs():
    """Every `site-packages/nvidia/*/bin` directory that exists."""
    dirs = []
    roots = list(site.getsitepackages())
    if hasattr(site, "getusersitepackages"):
        roots.append(site.getusersitepackages())
    for sp in roots:
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            if os.path.isdir(d) and d not in dirs:
                dirs.append(d)
    return dirs


def enable_cuda():
    """Put the NVIDIA DLLs on PATH. Safe to call repeatedly."""
    global _enabled
    if _enabled:
        return True
    dirs = nvidia_dll_dirs()
    if not dirs:
        return False
    current = os.environ.get("PATH", "")
    missing = [d for d in dirs if d not in current]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing) + os.pathsep + current
    for d in dirs:                      # belt and braces for Python-side loads
        try:
            os.add_dll_directory(d)
        except (OSError, AttributeError):
            pass
    _enabled = True
    return True


def enable_ffmpeg():
    """Put a bundled ffmpeg on PATH.

    The rest of this project never needs ffmpeg -- libsndfile decodes and
    encodes MP3 on its own. `audio_separator` is the exception: it shells out to
    `ffmpeg -version` at construction time and refuses to start without it. The
    `static-ffmpeg` wheel ships real binaries, so this stays a pip dependency
    rather than a system install the user has to do by hand.
    """
    import shutil
    if shutil.which("ffmpeg"):
        return True
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        return shutil.which("ffmpeg") is not None
    except Exception:
        return False


def enable_for(model):
    """Prepare the GPU stack for this model's backend, and only that one.

    The two stacks cannot both be on PATH. onnxruntime needs the
    `nvidia-*-cu12` wheels in `site-packages/nvidia/*/bin`; torch ships its own
    copies of the same libraries in `torch/lib`. Put the nvidia wheels first
    and torch loads *their* cuDNN instead of its own, which fails at import:

        OSError: [WinError 127] ... Error loading cudnn_cnn64_9.dll

    -- from a torch that imports perfectly in a clean process. So the PATH
    surgery is scoped to ONNX models, and torch models are left alone to find
    the DLLs shipped alongside them.
    """
    if backend_for(model) == "onnxruntime":
        return enable_cuda()
    import torch  # noqa: F401  -- bind torch's own DLLs while PATH is clean
    return True


def backend_for(model):
    """Which runtime `audio_separator` will use for this model file.

    Model family is decided by the file extension, and it decides everything
    about speed. `.onnx` (MDX-Net) runs on onnxruntime. `.yaml` (Demucs) and
    `.pth`/`.th` (VR arch) run on torch. Two independent GPU stacks, and
    getting one working tells you nothing about the other.
    """
    ext = str(model).lower().rsplit(".", 1)[-1]
    return "onnxruntime" if ext == "onnx" else "torch"


def torch_status():
    """Whether the *torch* half of the stack can reach the GPU.

    This exists because of a costly wrong assumption: `cuda_status()` reported
    a genuinely working CUDA session, so separation was expected to be fast --
    but the default model here is `htdemucs_ft.yaml`, a Demucs model, and
    Demucs runs on torch. torch was a `+cpu` build. onnxruntime was never
    asked to do anything.

    Worse, `htdemucs_ft` is the *fine-tuned* variant: a bag of four separate
    checkpoints, so it costs 4x a plain `htdemucs`. On CPU one four-minute
    track ran past nineteen minutes without finishing.

    Fix: install the CUDA build of the same torch version, so nothing else in
    the environment sees a different API --
        pip install --index-url https://download.pytorch.org/whl/cu126 \\
            torch==2.13.0+cu126
    """
    try:
        import torch
    except ImportError as e:
        return {"available": False, "reason": f"import failed: {e}"}
    info = {"torch": torch.__version__,
            "built_with_cuda": torch.version.cuda}
    try:
        ok = torch.cuda.is_available()
        info["available"] = bool(ok)
        if ok:
            info["device"] = torch.cuda.get_device_name(0)
        else:
            info["reason"] = ("CPU-only build" if "+cpu" in torch.__version__
                              else "no usable CUDA device")
    except Exception as e:
        info.update(available=False, reason=f"{type(e).__name__}: {e}")
    return info


def device_status(model=None):
    """GPU readiness of the backend that actually matters for `model`.

    Deliberately does not probe both backends when a model is given: probing
    onnxruntime is what puts the nvidia wheels on PATH, and doing that before a
    torch model runs is exactly the collision `enable_for` avoids.
    """
    if model is None:
        return {"onnxruntime": cuda_status(), "torch": torch_status()}
    backend = backend_for(model)
    enable_for(model)
    both = {backend: cuda_status() if backend == "onnxruntime"
            else torch_status()}
    return {"model": str(model), "backend": backend,
            "gpu": both[backend].get("available", False), "detail": both[backend]}


def cuda_status():
    """Prove CUDA is usable by opening a real session, not by reading a list.

    `get_available_providers()` reports what the onnxruntime *build* supports,
    not what this machine can run -- it listed CUDAExecutionProvider on a box
    where the CUDA DLLs were absent.
    """
    enable_cuda()
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper
    except ImportError as e:
        return {"available": False, "reason": f"import failed: {e}"}

    info = {"onnxruntime": ort.__version__,
            "listed": ort.get_available_providers()}
    if "CUDAExecutionProvider" not in info["listed"]:
        info.update(available=False, reason="build has no CUDA provider")
        return info

    n = 256
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "B"], ["C"])], "probe",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [n, n]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [n, n])],
        [helper.make_tensor_value_info("C", TensorProto.FLOAT, [n, n])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 9
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(model.SerializeToString(), so,
                                    providers=["CUDAExecutionProvider"])
        used = sess.get_providers()
        x = np.random.rand(n, n).astype(np.float32)
        sess.run(None, {"A": x, "B": x})
        info["used"] = used
        info["available"] = "CUDAExecutionProvider" in used
        if not info["available"]:
            info["reason"] = "silently fell back to CPU"
    except Exception as e:
        info.update(available=False, reason=f"{type(e).__name__}: {e}")
    return info
