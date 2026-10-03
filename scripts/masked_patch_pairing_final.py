"""Final masked-patch pairing: cashew -> expanded corpus.

Corpus rules (all decided by inspection; revised 2026-09-23):
  pipe_fryum   all classes
  fire_hood    defect_type=pit                      -> holes   (the hole donor since 2026-09-23)
  fire_hood    contamination w/ burn-like captions  -> burnt
  multi-label and stuck/other dropped on both sides.

  Dropped 2026-09-23:
    fryum      pipe_fryum alone covers every class it served (scratches 18/15, burnt 15/10,
               colour_same 20/10, colour_diff 10/9 supply/demand), and its morphology is a worse match.
    hazelnut   its "holes" are craters 31x the area of a cashew hole and not darker than the shell
               (dL* +0.1 vs the cashew target -21.6); fire_hood pits are 321 px / dL* -4.8, i.e. the
               same size class as the cashew holes and painted at native scale.

Usage: python scripts/masked_patch_pairing_final.py
"""
import csv, json, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
V = ROOT / "anomverse_extension/datasets/VisA_validation_dataset"
REST = V / "restVisA/VisA_20220922"
D = ROOT / "anomverse_extension/datasets/full_training_dataset"
OUT = ROOT / "results/masked_patch_knn"; OUT.mkdir(parents=True, exist_ok=True)
MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
SIZE, PATCH = 224, 14
GRID = SIZE // PATCH
REUSE_CAP = 2

CANON = {
    "small holes": "holes",
    "corner or edge breakage": "breakage", "corner and edge breakage": "breakage",
    "middle breakage": "breakage", "small cracks": "breakage",
    "small scratches": "scratches", "burnt": "burnt",
    "same colour spot": "colour_same", "similar colour spot": "colour_same",
    "different colour spot": "colour_diff",
    "stuck together": "stuck", "fryum stuck together": "stuck", "other": "other",
}
DROP = {"stuck", "other"}
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
BURN = re.compile(r"\b(burn\w*|scorch\w*|charred|char|sooty|soot|blacken\w*|singe\w*)\b", re.I)

def canon(l): return {CANON.get(a.strip().lower(), "?") for a in l.split(",") if a.strip()}
def n_raw(l): return len([a for a in l.split(",") if a.strip()])
def anno(p):
    m = {}
    with open(p) as fh:
        for r in csv.reader(fh):
            if len(r) > 1 and "Anomaly" in r[0]: m[Path(r[0]).stem] = r[1]
    return m

def visa_items(obj, img_dirs, mask_dir, annop, exclude=()):
    out = []
    for i, raw in sorted(anno(annop).items()):
        c = canon(raw)
        if (c & DROP) or not c or n_raw(raw) > 1: continue
        cls = sorted(c)[0]
        if cls in exclude: continue
        img = next((dd / f"{i}{e}" for dd in img_dirs for e in (".JPG", ".jpg", ".png")
                    if (dd / f"{i}{e}").exists()), None)
        mask = next((mask_dir / f"{i}{e}" for e in (".png", ".PNG")
                     if (mask_dir / f"{i}{e}").exists()), None)
        if img and mask:
            out.append({"src": obj, "id": i, "img": img, "mask": mask, "raw": raw, "cls": cls})
    return out

def train_items(src, cls, pred):
    caps = train_items._caps
    out = []
    for m in train_items._master:
        if not pred(m): continue
        mp = m.get("mask_path")
        if not mp or not (D / mp).exists(): continue
        out.append({"src": src, "id": Path(m["image_path"]).stem, "img": D / m["image_path"],
                    "mask": D / mp, "raw": f"{src}/{m.get('defect_type')}", "cls": cls,
                    "cap": caps.get(m["image_path"], "")})
    return out

def to_tensor(p):
    t = torch.from_numpy(np.array(p)).permute(2, 0, 1).float() / 255.0
    return (t - CLIP_MEAN) / CLIP_STD

def weights(mk, dilate=4):
    m = torch.from_numpy((np.array(mk) > 0).astype(np.float32))[None, None]
    if dilate:
        k = 2*dilate + 1
        m = (F.max_pool2d(m, kernel_size=k, stride=1, padding=dilate) > 0).float()
    return F.avg_pool2d(m, kernel_size=PATCH).flatten()

def crop_view(ip, mp, pad=0.15):
    im = Image.open(ip).convert("RGB"); mk = Image.open(mp).convert("L")
    if mk.size != im.size: mk = mk.resize(im.size, Image.NEAREST)
    a = np.array(mk) > 0
    if not a.any(): return None, None
    ys, xs = np.where(a); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = max(int(max(y1-y0+1, x1-x0+1) * (1 + 2*pad)), SIZE)
    cy, cx = (y0+y1)//2, (x0+x1)//2
    left = int(np.clip(cx-side//2, 0, max(0, im.width-side)))
    top = int(np.clip(cy-side//2, 0, max(0, im.height-side)))
    b = (left, top, min(left+side, im.width), min(top+side, im.height))
    return im.crop(b).resize((SIZE, SIZE), Image.LANCZOS), mk.crop(b).resize((SIZE, SIZE), Image.NEAREST)

def full_view(ip, mp):
    return (Image.open(ip).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS),
            Image.open(mp).convert("L").resize((SIZE, SIZE), Image.NEAREST))

@torch.no_grad()
def embed(items, enc, dev, bs=16):
    acc = defaultdict(list)
    for s in range(0, len(items), bs):
        ch = items[s:s+bs]
        for name, b in (("crop", crop_view), ("full", full_view)):
            px, w = [], []
            for it in ch:
                ci, cm = b(it["img"], it["mask"])
                if ci is None:
                    px.append(torch.zeros(3, SIZE, SIZE)); w.append(torch.zeros(GRID*GRID)); continue
                px.append(to_tensor(ci)); w.append(weights(cm))
            o = enc(pixel_values=torch.stack(px).to(dev), output_hidden_states=True)
            tok = o.hidden_states[-2][:, 1:, :].float()
            W = torch.stack(w).to(dev); W2 = W.clone(); W2[W2.sum(1) == 0] = 1.0
            pooled = (tok * W2[..., None]).sum(1) / W2.sum(1, keepdim=True).clamp(min=1e-6)
            acc[name].append(F.normalize(pooled, dim=-1).cpu())
    return {k: torch.cat(v) for k, v in acc.items()}

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_items._caps = {c["image_path"]: c["caption"] for c in
                         json.loads((D / "captions_from_master.json").read_text(encoding="utf-8"))}
    train_items._master = json.loads((D / "master_training.json").read_text(encoding="utf-8"))

    q = visa_items("cashew",
                   [V/"datasets/easy_test/cashew/Data/Images/Anomaly/easy_imgs",
                    V/"datasets/easy_test/cashew/Data/Images/Anomaly/hard_imgs"],
                   V/"datasets/easy_test/cashew/Data/Masks/Anomaly",
                   V/"datasets/easy_test/cashew/image_anno.csv")

    c  = visa_items("pipe_fryum", [REST/"pipe_fryum/Data/Images/Anomaly"],
                    REST/"pipe_fryum/Data/Masks/Anomaly", REST/"pipe_fryum/image_anno.csv")
    c += train_items("fire_hood", "holes", lambda m: "fire_hood" in m["image_path"]
                     and (m.get("defect_type") or "").lower() == "pit")
    c += train_items("fire_hood", "burnt", lambda m: "fire_hood" in m["image_path"]
                     and (m.get("defect_type") or "").lower() == "contamination"
                     and BURN.search(train_items._caps.get(m["image_path"], "")))

    print(f"queries {len(q)}   corpus {len(c)}  {dict(Counter(x['src'] for x in c))}")
    print("\nclass supply:")
    dem, sup = Counter(x["cls"] for x in q), Counter(x["cls"] for x in c)
    for k in sorted(set(dem) | set(sup)):
        print(f"  {k:14s} cashew {dem.get(k,0):3d}   corpus {sup.get(k,0):3d}")

    from transformers import CLIPVisionModelWithProjection
    enc = CLIPVisionModelWithProjection.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev).eval()
    Q, C = embed(q, enc, dev), embed(c, enc, dev)
    S = (((Q["crop"] @ C["crop"].T) + (Q["full"] @ C["full"].T)) / 2).numpy()

    free = np.mean([q[i]["cls"] == c[int(S[i].argmax())]["cls"] for i in range(len(q))])
    print(f"\nfree 1-NN class agreement: {free:.1%}")

    # HARD CONSTRAINT: only same canonical class may pair.
    MASK = np.array([[q[i]["cls"] == c[j]["cls"] for j in range(len(c))]
                     for i in range(len(q))])
    print(f"queries with >=1 same-class candidate: {int((MASK.sum(1) > 0).sum())}/{len(q)}")
    BIG = 1e6
    ri, ci = linear_sum_assignment(np.where(MASK, -S, BIG))
    assign = {int(i): int(j) for i, j in zip(ri, ci) if MASK[i, j]}
    used = Counter(assign.values())
    left = [i for i in range(len(q)) if i not in assign]
    for i in sorted(left, key=lambda i: -(S[i][MASK[i]].max() if MASK[i].any() else -9)):
        cand = [(S[i, j], j) for j in range(len(c)) if MASK[i, j] and used[j] < REUSE_CAP]
        if cand:
            sv, j = max(cand); assign[i] = j; used[j] += 1
    rows = []
    for i, j in sorted(assign.items()):
        rows.append({"id": q[i]["id"], "raw": q[i]["raw"], "cls": q[i]["cls"], "sim": float(S[i, j]),
                     "nn_src": c[j]["src"], "nn_id": c[j]["id"], "nn_raw": c[j]["raw"],
                     "nn_cls": c[j]["cls"], "hit": q[i]["cls"] == c[j]["cls"],
                     "img": str(q[i]["img"]), "nn_img": str(c[j]["img"])})
    rows.sort(key=lambda r: r["id"])
    hits = sum(r["hit"] for r in rows)
    print(f"1:1 Hungarian (+cap {REUSE_CAP} top-up): {hits}/{len(rows)} = {hits/len(rows):.1%}")
    print(f"sources chosen: {dict(Counter(r['nn_src'] for r in rows))}")
    per = defaultdict(lambda: [0, 0])
    for r in rows: per[r["cls"]][0] += r["hit"]; per[r["cls"]][1] += 1
    print("\nper class:")
    for k, (h, n) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f"  {k:14s}{h:3d}/{n:<3d} {h/n:>6.0%}")
    conf = Counter((r["cls"], r["nn_cls"]) for r in rows if not r["hit"])
    if conf:
        print("\nconfusions:")
        for (a, b), n in conf.most_common(8): print(f"  {n:3d}  {a} -> {b}")
    (OUT / "pairing_constrained.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # ── sheet ──
    T, GAP, PADX, COLS, LBL, HDR = 150, 3, 11, 6, 48, 32
    def font(s, b=False):
        for n in (("arialbd.ttf" if b else "arial.ttf"), "segoeui.ttf"):
            try: return ImageFont.truetype(n, s)
            except Exception: pass
        return ImageFont.load_default()
    F_, FB, FH = font(10), font(11, True), font(16, True)
    def sq(p, s=T):
        im = Image.open(p).convert("RGB"); w, h = im.size; m = min(w, h)
        return im.crop(((w-m)//2, (h-m)//2, (w-m)//2+m, (h-m)//2+m)).resize((s, s), Image.LANCZOS)
    def wrap(dr, t, f, mw, ml=2):
        o, cur = [], ""
        for w in t.split():
            x = (cur+" "+w).strip()
            if dr.textlength(x, font=f) <= mw: cur = x
            else:
                o.append(cur); cur = w
                if len(o) == ml: break
        if cur and len(o) < ml: o.append(cur)
        return o[:ml]
    groups = defaultdict(list)
    for r in rows: groups[r["cls"]].append(r)
    order = sorted(groups, key=lambda k: -len(groups[k]))
    PW, CW, CH = T*2+GAP, T*2+GAP+PADX, T+LBL+5
    W = COLS*CW+PADX
    H = 58 + sum(HDR + ((len(groups[g])+COLS-1)//COLS)*CH + 8 for g in order)
    sh = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(sh)
    dr.rectangle([0, 0, W, 54], fill="#0F1416")
    dr.text((PADX, 5), f"CLASS-CONSTRAINED  —  {len(rows)} cashews  |  masked-patch kNN within class only",
            font=FH, fill="white")
    dr.text((PADX, 25), "pipe_fryum + fryum(no breakage) + fire_hood(burnt) + hazelnut(holes, cut->scratches)",
            font=F_, fill="#8A9C98")
    dr.text((PADX, 38), f"green = partner shares the class ({hits}/{len(rows)})   red = mismatch",
            font=F_, fill="#8A9C98")
    y = 58
    for g in order:
        grp = sorted(groups[g], key=lambda r: -r["sim"]); gh = sum(r["hit"] for r in grp)
        dr.rectangle([0, y, W, y+HDR-5], fill="#243033")
        dr.text((PADX, y+5), f"{g}   n={len(grp)}   agreement {gh}/{len(grp)}", font=FH, fill="white")
        y += HDR
        for k, r in enumerate(grp):
            cx, cy = PADX+(k % COLS)*CW, y+(k//COLS)*CH
            col = "#1F6B45" if r["hit"] else "#9E2A3C"
            try:
                sh.paste(sq(Path(r["img"])), (cx, cy)); sh.paste(sq(Path(r["nn_img"])), (cx+T+GAP, cy))
            except Exception: continue
            dr.rectangle([cx, cy, cx+PW, cy+T], outline=col, width=3)
            dr.line([cx+T+GAP//2, cy, cx+T+GAP//2, cy+T], fill=col, width=1)
            dr.text((cx+PW-38, cy+3), f"{r['sim']:.3f}", font=FB, fill="white")
            dr.text((cx+1, cy+T+2), f"cashew {r['id']}", font=FB, fill="#111")
            yy = cy+T+15
            for ln in wrap(dr, r["raw"], F_, T-2): dr.text((cx+1, yy), ln, font=F_, fill=col); yy += 12
            rx = cx+T+GAP
            dr.text((rx+1, cy+T+2), f"{r['nn_src']} {r['nn_id']}", font=FB, fill="#111")
            yy = cy+T+15
            for ln in wrap(dr, r["nn_raw"], F_, T-2): dr.text((rx+1, yy), ln, font=F_, fill="#555"); yy += 12
        y += ((len(grp)+COLS-1)//COLS)*CH + 8
    f = OUT / "pairing_constrained.png"
    sh.save(f, optimize=True)
    print(f"\nsaved {f} ({sh.width}x{sh.height}) {f.stat().st_size/1024/1024:.1f} MB")

if __name__ == "__main__":
    main()
