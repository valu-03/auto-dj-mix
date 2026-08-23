"""Loading, inspecting and converting audio."""

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

ANALYSIS_RATE = 22050
RENDER_RATE = 44100


def describe(path):
    """Read a file's header without decoding the audio."""
    info = sf.info(str(path))
    return {
        "path": Path(path),
        "name": Path(path).stem,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration": info.duration,
        "format": info.format,
        "subtype": info.subtype,
    }


BLOCK = 65536


def _read_forgiving(path):
    """Decode a file, keeping whatever survives if it is damaged."""
    try:
        return sf.read(str(path), dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        pass

    chunks = []
    with sf.SoundFile(str(path)) as fh:
        rate = fh.samplerate
        try:
            while True:
                block = fh.read(BLOCK, dtype="float32", always_2d=True)
                if block.shape[0] == 0:
                    break
                chunks.append(block)
        except sf.LibsndfileError:
            pass

    if not chunks:
        raise RuntimeError(f"no decodable audio in {path}")
    return np.concatenate(chunks, axis=0), rate


def load(path, sample_rate=None, mono=False):
    """Load audio as float32, shaped (channels, frames)."""
    audio, file_rate = _read_forgiving(path)
    audio = audio.T

    if mono and audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)

    if sample_rate is not None and sample_rate != file_rate:
        audio = soxr.resample(audio.T, file_rate, sample_rate).T
        file_rate = sample_rate

    return np.ascontiguousarray(audio), file_rate


def to_samples(seconds, sample_rate):
    """Convert a time in seconds to a sample index."""
    return int(round(seconds * sample_rate))


def to_seconds(samples, sample_rate):
    """Convert a sample index to a time in seconds."""
    return samples / sample_rate


def duration(audio, sample_rate):
    """True duration in seconds, from the decoded array."""
    return audio.shape[1] / sample_rate


def clip(audio, sample_rate, start, end):
    """Slice by seconds, clamped to the real bounds of the array."""
    a = max(0, to_samples(start, sample_rate))
    b = min(audio.shape[1], to_samples(end, sample_rate))
    return audio[:, a:max(a, b)]


MP3_KBPS = 320          # LAME's maximum; the sources go up to 320 too


def _encode_mp3_lame(path, audio, sample_rate, kbps=MP3_KBPS):
    """Encode via LAME, through the ffmpeg that ships with static-ffmpeg.

    libsndfile can write MP3, but not well: its exposed compression level tops
    out at about 154 kbps even at the best setting, and the scale is inverted
    (0.0 is the *best* quality, 0.7 gives 56 kbps). Since the best sources here
    are already 320 kbps, re-encoding the master at 154 throws away quality
    that the material actually had.

    ffmpeg is already a dependency for stem separation, so LAME is available
    at no extra cost. Raw float32 goes in over the pipe -- no intermediate WAV,
    no second quantisation.
    """
    import shutil
    import subprocess
    from .stems import cuda
    cuda.enable_ffmpeg()
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "f32le", "-ar", str(int(sample_rate)),
           "-ac", str(int(audio.shape[0])), "-i", "pipe:0",
           "-codec:a", "libmp3lame", "-b:a", f"{int(kbps)}k",
           str(path)]
    try:
        p = subprocess.run(cmd, input=np.ascontiguousarray(audio.T,
                                                           dtype=np.float32).tobytes(),
                           capture_output=True)
    except OSError:
        return False
    if p.returncode != 0:
        return False
    return Path(path).exists() and Path(path).stat().st_size > 0


def save(path, audio, sample_rate, subtype=None, quality=None, kbps=MP3_KBPS):
    """Write a (channels, frames) float32 array to disk.

    Picks the format from the extension. WAV masters are written as **24-bit**,
    not 16: this file is a master that gets re-encoded to MP3 afterwards, and
    quantising to 16 bits first adds a noise floor the encoder then has to
    spend bits on. 24-bit costs 50% more disk and nothing else.
    """
    audio = np.atleast_2d(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    path = Path(path)
    is_mp3 = path.suffix.lower() == ".mp3"

    if is_mp3:
        if _encode_mp3_lame(path, audio, sample_rate, kbps):
            return path
        # Fallback: libsndfile. Note the scale is INVERTED -- 0.0 is best --
        # and `set_compression_level` does not exist on SoundFile (the private
        # `_set_compression_level` does), which is why the old `try/except:
        # pass` silently never applied it at all.
        with sf.SoundFile(str(path), "w", samplerate=sample_rate,
                          channels=audio.shape[0], format="MP3",
                          subtype=subtype or "MPEG_LAYER_III") as fh:
            if quality is not None:
                fh._set_compression_level(float(np.clip(quality, 0.0, 0.99)))
            fh.write(audio.T)
        return path

    sf.write(str(path), audio.T, sample_rate, subtype=subtype or "PCM_24")
    return path