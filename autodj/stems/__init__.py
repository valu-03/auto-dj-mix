"""Stem separation: splitting a track by SOURCE rather than by frequency.

Importing this package puts a bundled ffmpeg on PATH, because audio_separator
refuses to construct without one.

It deliberately does NOT touch the CUDA search path any more. Which GPU stack
to prepare depends on the model -- ONNX models need the nvidia wheels on PATH,
torch models must not have them there -- so that decision is made per model, in
`cuda.enable_for()`, once the model is known.
"""

from .cuda import (backend_for, cuda_status, device_status,  # noqa: F401
                   enable_cuda, enable_ffmpeg, enable_for, torch_status)

enable_ffmpeg()
