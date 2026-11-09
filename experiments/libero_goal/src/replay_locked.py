"""Replay the archived SmolVLA numerical edits without making new SDK calls."""
import argparse
from pathlib import Path
import subprocess
import sys

SRC = Path(__file__).resolve().parent
RESULTS = SRC.parent / "results" / "repair-series"
ARMS = {
    "sequential-repair": ("v0", "v1", "v2", "v3"),
    "bp-control": ("stock", "bp", "wsp"),
    "activation-repair": ("stock", "auto-bp", "astra-bp", "astra-wsp"),
}


def launch(script, *args):
    subprocess.run([sys.executable, str(SRC / script), *(str(x) for x in args)], check=True)


def direct(out, gpu):
    root = RESULTS / "weight-repair"
    common = ("--screen", RESULTS / "replay-weight-screen", "--render-gpu", gpu)
    launch("weight-repair/repair.py", "--out", out / "stock", *common)
    launch("weight-repair/repair.py", "--out", out / "edits", *common,
           "--patch", root / "conservative.json", "--patch", root / "assertive.json")
    launch("weight-repair/settle_verify.py", "--out", out / "settling",
           "--case", out / "edits/conservative/fault/init-002-bottle",
           "--patch", root / "conservative.json", "--render-gpu", gpu)


def multitask(out, gpu):
    root = SRC / "multitask-repair"
    names = ("stock", "bottle", "bowl", "merged")
    partitions = ("historical-clean", "bowl-development", "heldout")
    for name in partitions:
        launch("multitask-repair/matrix.py", "--cases", root / f"{name}.json",
               "--out", out / name, "--a", RESULTS / "weight-repair/conservative.json",
               "--b", RESULTS / "multitask-repair/bowl-locked.json", "--arms", *names,
               "--render-gpu", gpu)
    launch("multitask-repair/verify_matrix.py", "--out", out / "verified-matrix.json",
           *(f"{out / name}:{','.join(names)}" for name in partitions))
    launch("multitask-repair/extend_retention.py", "--out", out / "retention",
           "--render-gpu", gpu, out / "bowl-development/merged/bowl-rim-dev",
           out / "heldout/bottle/new-bowl-7")


def fixed(series, out, gpu, arm):
    default_arms = ARMS[series]
    for version in ([arm] if arm else default_arms):
        model = ()
        if version not in ("stock", "v0"):
            edit = RESULTS / series / f"{version}.json"
            if series == "sequential-repair" and version == "v3":
                checkpoint = out / "checkpoint-joint"
                launch("sequential-repair/materialize.py", "--edits", edit, "--out", checkpoint)
                model = ("--checkpoint", checkpoint, "--vlm-assets", checkpoint / "vlm_assets")
            else:
                model = ("--edits", edit)
        for split in ("dev", "preserve", "heldout"):
            dirname = version if series == "sequential-repair" and split == "dev" else f"{version}-{split}"
            launch("sequential-repair/stage.py", out / dirname, *model,
                   "--split", split, "--heldout", SRC / series / "heldout.json", "--render-gpu", gpu)
    if arm is None:
        launch(f"{series}/score.py", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("series", choices=("weight-repair", "multitask-repair", *ARMS))
    p.add_argument("--out", type=Path, required=True, help="new directory for raw rollouts")
    p.add_argument("--render-gpu", type=int, default=0)
    p.add_argument("--arm", help="for sequential/BP/localization only, replay one arm instead of the full matrix")
    a = p.parse_args()
    if a.arm and a.arm not in ARMS.get(a.series, ()):
        p.error(f"Invalid --arm for {a.series}: {a.arm}")
    out = a.out.resolve()
    out.mkdir(parents=True)
    if a.series == "weight-repair":
        direct(out, a.render_gpu)
    elif a.series == "multitask-repair":
        multitask(out, a.render_gpu)
    else:
        fixed(a.series, out, a.render_gpu, a.arm)


if __name__ == "__main__":
    main()
