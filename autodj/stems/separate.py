"""Splitting a track into vocals / drums / bass / other, with a disk cache.

Never run this across a library. 686 tracks x 4 stems is hundreds of gigabytes
and hours even on a GPU. Stems are computed on demand for the handful of tracks
in a setlist, kept as FLAC, and evicted oldest-first.
"""

import os
import sys
import re
import shutil
import time
from pathlib import Path

from . import cuda

cuda.enable_ffmpeg()   # CUDA is prepared per model, in check_device()

STEM_NAMES = ("vocals", "drums", "bass", "other")
DEFAULT_MODEL = "htdemucs_ft.yaml"      # 4-stem Demucs, best quality/speed here
CACHE_DIR = Path("cache/stems")
MODEL_DIR = Path("cache/models")
MAX_CACHE_GB = 20.0

_separator = None
_loaded_model = None
_patched = False


def _patch_librosa():
    """Let audio_separator call a librosa keyword that no longer exists.

    `librosa.get_duration(filename=...)` was renamed to `path=` in 0.10 and
    deleted in 1.0. audio_separator 0.44 still uses the old name, in
    `write_audio` -- so separation runs to completion on the GPU and then
    throws on the very last step, writing nothing. Losing a minute of GPU work
    to a keyword rename.

    Downgrading librosa is not an option: every analyser in this project is
    built on the 1.0 API. So the shim goes here, accepting the old name and
    forwarding it. Harmless to our own calls, which pass `path=` already.
    """
    global _patched
    if _patched:
        return
    import librosa
    original = librosa.get_duration

    def get_duration(*args, filename=None, **kw):
        if filename is not None:
            kw["path"] = filename
        return original(*args, **kw)

    librosa.get_duration = get_duration
    _patched = True


CHUNK = 1 << 20         # 1 MB from each end is plenty to separate two files


def _key(path, chunk=CHUNK):
    """Cache key from the file's CONTENT, not its path or its mtime.

    Analysis is keyed on path+size+mtime, which is right for it: an analysis is
    cheap to redo and the key has to be cheap to compute for a whole library.
    Separation is neither. A minute of GPU time per track is not something to
    throw away because a file was copied, moved, or restored from a backup --
    and that is exactly what happened here: four tracks were replaced with
    identical copies, every mtime changed, and six separated tracks became
    unreachable while still sitting on disk.

    The output depends only on the audio bytes, so the key should too. Size
    plus a megabyte from each end is cheap on a ten-megabyte MP3 and is not
    going to collide with another track in any real library -- two files that
    match on all three are the same recording.
    """
    import hashlib
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha1(str(size).encode("utf-8"))
    with open(p, "rb") as fh:
        h.update(fh.read(chunk))
        if size > 2 * chunk:
            fh.seek(-chunk, 2)
            h.update(fh.read(chunk))
    return h.hexdigest()[:16]


STAGE_DIR = Path("cache/stems/_staging")


def _get_separator(model=DEFAULT_MODEL):
    """One Separator instance, reused. Loading a model costs seconds.

    Always writes to STAGE_DIR, never to the per-track cache directory.
    Reassigning `Separator.output_dir` between calls does NOT work: the loaded
    model instance keeps the directory it was constructed with, so the second
    track in a loop wrote its four stems into the *first* track's cache folder.
    That folder then held eight files, `load_cached` matched the `(Tag)` on the
    wrong ones, and the second track looked like it had produced nothing --
    while the first silently started returning someone else's audio.

    Staging then moving sidesteps the whole question of which directory the
    library thinks it is writing to.
    """
    global _separator, _loaded_model
    _patch_librosa()
    from audio_separator.separator import Separator

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    if _separator is None:
        _separator = Separator(
            output_dir=str(STAGE_DIR),
            output_format="FLAC",
            model_file_dir=str(MODEL_DIR),
            use_autocast=True,
        )
    if _loaded_model != model:
        _separator.load_model(model_filename=model)
        _loaded_model = model
    return _separator


def cache_dir_for(path, cache_dir=CACHE_DIR):
    return Path(cache_dir) / _key(path)


def is_cached(path, cache_dir=CACHE_DIR):
    d = cache_dir_for(path, cache_dir)
    return d.is_dir() and len(list(d.glob("*.flac"))) >= 4


def load_cached(path, cache_dir=CACHE_DIR):
    """Map stem name -> file path, or None if not separated yet."""
    d = cache_dir_for(path, cache_dir)
    if not d.is_dir():
        return None
    found = {}
    for f in d.glob("*.flac"):
        name = stem_name_of(f)
        if name:
            found[name] = f
    return found if len(found) >= 4 else None


def stem_name_of(path):
    """Which stem a separated file holds, read from its `(Tag)` only.

    Not a substring search over the filename. audio_separator names its output
    `<original stem>_(Vocals)_<model>.flac`, and the original title is in
    there -- so matching 'other' anywhere in the name silently claimed
    "Whigfield - An(other) Day_(Vocals)" as the *other* stem, overwriting the
    real one. Four correct files on disk, two of them mapped to the same array,
    and a mashup that quietly uses the vocal twice.
    """
    tags = re.findall(r"\(([^)]+)\)", Path(path).stem)
    for tag in reversed(tags):
        low = tag.strip().lower()
        if low in STEM_NAMES:
            return low
    return None


def check_device(model=DEFAULT_MODEL, require_gpu=True):
    """Refuse to start a separation that would silently crawl on the CPU.

    Neither backend raises when it cannot reach the GPU -- both just run, at
    roughly 1/40th speed. On a four-minute track that is the difference between
    half a minute and half an hour, with nothing in the output to say why. So
    the check happens once, up front, and says which backend it checked.
    """
    st = cuda.device_status(model)
    if st["gpu"] or not require_gpu:
        return st
    raise RuntimeError(
        f"{model} runs on {st['backend']}, which cannot reach the GPU "
        f"({st['detail'].get('reason', 'unknown')}). Separation would take "
        f"~40x longer. Pass require_gpu=False to run on CPU anyway.")


def _demucs_available():
    try:
        import demucs.apply          # noqa: F401
        import demucs.pretrained     # noqa: F401
        return True
    except ImportError:
        return False


def _separate_demucs(path, out, model=DEFAULT_MODEL, require_gpu=True):
    """Run Demucs in its own interpreter and collect what it wrote.

    Same model as the wrapped path, one less abstraction, and a device we set
    explicitly rather than infer from a model file's extension.

    A subprocess because `audio_separator` puts its vendored copy of Demucs on
    `sys.path[0]` when it loads a model, which shadows the installed package
    for the rest of the process -- see `_demucs_worker` for the detail. The two
    backends genuinely cannot share an interpreter.
    """
    import subprocess

    cmd = [sys.executable, "-m", "autodj.stems._demucs_worker",
           str(path), str(out), str(model), "1" if require_gpu else "0"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parents[2]))
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError(
            "demucs backend failed"
            + (f": {detail[-1]}" if detail else
               f" with exit code {proc.returncode}"))
    return proc.stdout.strip()


def separate(path, model=DEFAULT_MODEL, cache_dir=CACHE_DIR, force=False,
             require_gpu=True, backend=None):
    """Separate one track. Returns {stem_name: Path} and seconds taken.

    `backend` picks the engine: "audio-separator" (default) or "demucs". It
    falls back to the environment variable `AUTODJ_STEM_BACKEND` so the choice
    can be made without touching a call site, and falls back again to the
    wrapped path if demucs is not installed -- an unavailable backend should
    degrade, not raise, because the two produce the same stems.
    """
    path = Path(path)
    cached = None if force else load_cached(path, cache_dir)
    if cached:
        return cached, 0.0

    backend = (backend or os.environ.get("AUTODJ_STEM_BACKEND")
               or "audio-separator").lower()
    if backend == "demucs" and not _demucs_available():
        backend = "audio-separator"

    if backend == "demucs":
        out = cache_dir_for(path, cache_dir)
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        _separate_demucs(path, out, model, require_gpu)
        elapsed = time.time() - t0
        result = load_cached(path, cache_dir)
        if not result:
            raise RuntimeError(
                f"demucs produced no usable stems for {path.name}: "
                f"{sorted(p.name for p in out.iterdir())}")
        prune(cache_dir)
        return result, elapsed

    check_device(model, require_gpu)

    out = cache_dir_for(path, cache_dir)
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # Staging is emptied first so anything found afterwards is certainly ours.
    shutil.rmtree(STAGE_DIR, ignore_errors=True)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    sep = _get_separator(model)
    sep.separate(str(path))
    elapsed = time.time() - t0

    produced = [f for f in STAGE_DIR.glob("*.flac") if stem_name_of(f)]
    if len(produced) < 4:
        raise RuntimeError(
            f"separation produced {len(produced)} stems for {path.name}: "
            f"{sorted(p.name for p in STAGE_DIR.iterdir())}")
    for f in produced:
        shutil.move(str(f), str(out / f.name))
    shutil.rmtree(STAGE_DIR, ignore_errors=True)

    result = load_cached(path, cache_dir)
    if not result:
        raise RuntimeError(f"separation produced no stems for {path.name}: "
                           f"{sorted(p.name for p in out.iterdir())}")
    prune(cache_dir)
    return result, elapsed


def cache_size_gb(cache_dir=CACHE_DIR):
    d = Path(cache_dir)
    if not d.is_dir():
        return 0.0
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9


def prune(cache_dir=CACHE_DIR, max_gb=MAX_CACHE_GB):
    """Evict oldest stem sets until the cache fits."""
    d = Path(cache_dir)
    if not d.is_dir():
        return 0
    dirs = sorted((p for p in d.iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime)
    removed = 0
    while cache_size_gb(cache_dir) > max_gb and len(dirs) > 1:
        shutil.rmtree(dirs.pop(0), ignore_errors=True)
        removed += 1
    return removed
