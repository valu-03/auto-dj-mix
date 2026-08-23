"""Scanning a music folder, reading tags, and caching analysis results."""

import hashlib
import json
from pathlib import Path

import mutagen
import numpy as np

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".aiff", ".aif"}
CACHE_DIR = Path("cache")
CACHE_VERSION = 6


def scan(folder, recursive=True):
    """Every audio file in a folder, sorted."""
    folder = Path(folder)
    walk = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in walk if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def tags(path):
    """Artist, title and year from ID3, falling back to the filename."""
    artist = title = year = None
    try:
        meta = mutagen.File(path, easy=True)
        if meta:
            artist = (meta.get("artist") or [None])[0]
            title = (meta.get("title") or [None])[0]
            raw = (meta.get("date") or meta.get("originaldate") or [None])[0]
            if raw:
                digits = "".join(c for c in str(raw)[:4] if c.isdigit())
                if len(digits) == 4 and 1900 < int(digits) < 2100:
                    year = int(digits)
    except Exception:
        pass

    stem = Path(path).stem
    # Strip a leading track number: "01 - Artist - Title" -> "Artist - Title"
    parts = [s.strip() for s in stem.split(" - ")]
    if len(parts) >= 2 and parts[0].isdigit():
        parts = parts[1:]

    if not artist and len(parts) >= 2:
        artist = parts[0]
    if not title:
        title = " - ".join(parts[1:]) if len(parts) >= 2 else parts[0]

    return {"artist": artist or "Unknown", "title": title or stem, "year": year}


def fingerprint(path):
    """Identity of a file's *content*, cheaply: size + mtime + cache version.

    Manual corrections join the key. They change what the analysis produces, so
    they have to change where it is stored -- otherwise saving a corrected
    downbeat returns the old cached answer and looks like the fix did nothing.
    Including them also means reverting a correction restores the previous
    cache entry instead of forcing a re-analysis.
    """
    from . import corrections
    p = Path(path)
    st = p.stat()
    raw = (f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}|v{CACHE_VERSION}"
           f"|{corrections.digest(p)}")
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _jsonable(obj):
    """json.dumps cannot handle numpy types. Convert them."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON-serialisable: {type(obj)}")


def cached(path, compute, cache_dir=CACHE_DIR, force=False):
    """Return cached analysis for a file, computing and storing it if absent."""
    store = Path(cache_dir)
    entry = store / f"{fingerprint(path)}.json"

    if entry.exists() and not force:
        try:
            return json.loads(entry.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    data = compute(path)

    store.mkdir(parents=True, exist_ok=True)
    tmp = entry.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=_jsonable, indent=1), encoding="utf-8")
    tmp.replace(entry)
    return data
