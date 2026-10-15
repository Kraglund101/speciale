#!/usr/bin/env python3
"""Drop-in for run_uninet_cashew.py: identical training, plus <output-dir>/image_scores.json = the per-image score and
label (+ image path) of every evaluate() call (epoch, test set: "main" or the shift level 0x..4x, aggregation). evaluate() is
re-compiled from its own source with ONE added line after the image-AUROC line, so every metric stays as it is.
Used through train_uninet.py with train_uninet.RUNNER pointed here (scripts/open_set_scores.py)."""
import inspect
import json
import sys
from pathlib import Path

RUNNER = (Path(__file__).resolve().parents[1] / "anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test"
          / "cashew/experiment_UniNet/temp_uninet_experiment/run_uninet_cashew.py")
sys.path.insert(0, str(RUNNER.parent)); sys.argv[0] = str(RUNNER)
import run_uninet_cashew as M  # noqa: E402

LOG: list[dict] = []
ANCHOR = "        img_auroc = roc_auc_score(all_labels_concat, img_scores)\n"


def _score_log(agg: str, labels, scores) -> None:
    fr = sys._getframe(1); dl = fr.f_locals["dataloader"]      # evaluate()
    while fr is not None and "test_loader" not in fr.f_locals:  # up to the training loop (past the no_grad wrapper)
        fr = fr.f_back
    f = fr.f_locals if fr is not None else {}
    names = {id(v): k for k, v in f.get("shift_loaders", {}).items()}; names[id(f.get("test_loader"))] = "main"
    smp = getattr(getattr(dl, "dataset", None), "samples", None)  # CashewDataset: (img, mask, label, canvas); test loaders are not shuffled
    imgs = [str(x[0]) for x in smp] if smp is not None else None
    assert imgs is None or (len(imgs) == len(labels) and [int(x[2]) for x in smp] == [int(x) for x in labels]), "image order"
    LOG.append({"epoch": f.get("epoch"), "shift": names.get(id(dl), "?"), "agg": agg, "images": imgs,
                "labels": [int(x) for x in labels], "scores": [float(x) for x in scores]})


src = inspect.getsource(M.evaluate); assert src.count(ANCHOR) == 1, "runner evaluate() changed: anchor line not found once"
M.__dict__["_score_log"] = _score_log
exec(compile(src.replace(ANCHOR, ANCHOR + "        _score_log(agg_name, all_labels_concat, img_scores)\n"), M.__file__, "exec"), M.__dict__)

if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    try:
        M.main()
    finally:
        out.mkdir(parents=True, exist_ok=True); json.dump(LOG, open(out / "image_scores.json", "w"))
