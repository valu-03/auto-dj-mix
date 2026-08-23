"""Run Demucs in a process of its own, and write the stems to a directory.

A separate process is not caution, it is necessity. `audio_separator` prepends
its own vendored copy of Demucs to `sys.path[0]` when it loads a model:

    sys.path[0] = .../audio_separator/separator/architectures/../uvr_lib_v5

From that moment `import demucs` resolves to the vendored fork for the rest of
the process, and `demucs.pretrained.get_model` goes looking for the fork's
`remote/files.txt` instead of the installed package's. The two backends
therefore cannot share an interpreter, and no amount of import ordering inside
one process fixes it -- the path is rewritten at model-load time, which is
after any import we control.

Since separation already costs about a minute per track and communicates
through files on disk, the cost of a new interpreter is not measurable.

    python -m autodj.stems._demucs_worker <track> <out_dir> <model> <gpu:0|1>
"""

import sys
from pathlib import Path


def main(argv):
    if len(argv) != 4:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    track, out_dir, model, want_gpu = argv
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    import demucs.apply
    import demucs.pretrained

    # If this fires, something imported the vendored fork first and the whole
    # point of the subprocess has been defeated.
    if "uvr_lib_v5" in str(Path(demucs.__file__)):
        print(f"demucs resolved to the vendored fork at {demucs.__file__}",
              file=sys.stderr)
        return 3

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from autodj import audio as audio_mod

    name = Path(model).stem
    net = demucs.pretrained.get_model(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if want_gpu == "1" and device != "cuda":
        print("CUDA is not available and the GPU was required",
              file=sys.stderr)
        return 4
    net.to(device).eval()

    wav, _rate = audio_mod.load(track, sample_rate=net.samplerate)
    a = np.atleast_2d(np.asarray(wav, dtype=np.float32))
    if a.shape[0] == 1:
        a = np.vstack([a[0], a[0]])

    # Demucs expects the input normalised to zero mean and unit variance, and
    # the sources scaled back afterwards. Skipping this does not fail -- it
    # quietly separates a signal at the wrong level and the stems no longer
    # sum to the source.
    ref = a.mean(0)
    mean, std = float(ref.mean()), float(ref.std() or 1.0)
    tensor = torch.from_numpy((a - mean) / std).to(device)

    with torch.no_grad():
        sources = demucs.apply.apply_model(net, tensor[None], device=device,
                                           progress=False, overlap=0.25)[0]
    sources = sources * std + mean

    title = Path(track).stem
    for source, chunk in zip(net.sources, sources):
        # The same `_(Tag)_` shape audio_separator writes, so `stem_name_of`
        # and `load_cached` cannot tell which backend produced a folder.
        f = out / f"{title}_({source.capitalize()})_{name}.flac"
        audio_mod.save(f, chunk.cpu().numpy(), net.samplerate)

    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"device={device} stems={len(net.sources)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
