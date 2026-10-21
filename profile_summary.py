#!/usr/bin/env python3
"""Summarise the py-spy profiles of the profiled seed (results/open_set_v2/profile_seed66/) -> PROFILE_SUMMARY.md.

py-spy raw format: one line per unique stack, "process PID:"cmd";...;process PID:"cmd";frame;frame;... count", sampled
at RATE Hz for every process separately (with --idle, so waiting time is included). Process-seconds = samples / RATE.
"""
import collections, re, sys
from pathlib import Path

D = Path(sys.argv[1]); RATE = 20
PROC = re.compile(r'^process (\d+):"(.*)"$')

# activity of a training process (run_uninet_cashew.py main), first match from the outermost frame inwards
TRAIN_ACT = [("discrepancy maps / figures", ("save_discrepancy_maps", "_plot_one_agg", "matplotlib", "savefig")),
             ("checkpoint save", ("save_checkpoint", "torch/serialization.py")),
             ("evaluation (clean + 5 shifts)", ("evaluate",)),
             ("training step", ("train_one_epoch",)),
             ("startup: data / model build", ("load_data", "build_model", "create_model", "make_model", "_swap_to_convnext",
                                               "DataLoader", "__init__")),
             ("startup: imports", ("_find_and_load",))]
WAIT = ("_next_data", "_get_data", "_try_get_data", "poll", "wait", "recv")


def kind(cmds: list[str]) -> str:
    deep = cmds[-1]
    if "run_uninet_cashew.py" in deep:
        return "trainer: data-loader worker" if sum("run_uninet_cashew.py" in c for c in cmds) > 1 else "trainer: main process"
    if any("run_uninet_cashew.py" in c for c in cmds[:-1]):
        return "trainer: data-loader worker"
    m = re.search(r"open_set_steps\.py (\w+)", deep)
    if m:
        return f"stage {m.group(1)}" + (": worker" if sum("open_set_steps.py" in c for c in cmds) > 1 else "")
    if any("open_set_steps.py" in c for c in cmds[:-1]):
        return "stage worker"
    if "train_uninet.py" in deep:
        return "train_uninet.py wrapper (waits for the trainer)"
    if "nvidia" in deep or "mps" in deep:
        return "nvidia/mps"
    return "launcher (open_set_launch.py)"


def load(f: Path):
    per_kind = collections.Counter(); acts = collections.Counter(); wait = collections.Counter(); pids = collections.defaultdict(set)
    func_incl = collections.Counter(); total = 0
    for line in f.read_text(errors="replace").splitlines():
        stack, _, n = line.rpartition(" ")
        if not n.isdigit():
            continue
        n = int(n); fr = stack.split(";"); cmds = []; pid = None; rest = []
        for x in fr:
            m = PROC.match(x)
            if m:
                cmds.append(m.group(2)); pid = m.group(1)
            else:
                rest.append(x)
        if not rest or rest[0].startswith(("_bootstrap ", "_bootstrap(")):   # secondary threads (pin-memory, queue feeders,
            continue                                                       # launcher pool) - main threads only = wall time
        k = kind(cmds or [""]); per_kind[k] += n; pids[k].add(pid); total += n
        if k == "trainer: main process":
            names = [x.split(" (")[0] + " " + x.split("(")[-1].split(":")[0] if "(" in x else x for x in rest]
            a = next((lab for lab, keys in TRAIN_ACT if any(any(kw in r for kw in keys) for r in rest)), "other")
            acts[a] += n
            if any(any(w in r for w in WAIT) for r in rest[-6:]):
                wait[a] += n
            for nm in set(names):
                func_incl[nm] += n
    return total, per_kind, pids, acts, wait, func_incl


def fmt_s(samples: int) -> str:
    s = samples / RATE
    return f"{s / 3600:.2f} h" if s >= 3600 else f"{s / 60:.1f} min" if s >= 60 else f"{s:.0f} s"


out = ["# Profile of one full seed (66) on the B200", ""]
tl = (D / "timeline.log").read_text() if (D / "timeline.log").exists() else ""
out += ["## Timeline", "```", tl.strip(), "```", ""]
lines = [l for l in (D.parent / "launch.log").read_text().splitlines() if "seed 66" in l and " fold " not in l]
out += ["## Stage wall times (launch.log)", "```", *lines[-12:], "```", ""]
runs = [l for l in (D.parent / "launch.log").read_text().splitlines() if "seed 66 fold" in l]
mins = [float(re.search(r"([\d.]+) min", l).group(1)) for l in runs if re.search(r"rc 0, ([\d.]+) min", l)]
if mins:
    out += [f"Trainings: {len(mins)} runs rc 0, wall per run mean {sum(mins) / len(mins):.1f} min "
            f"(min {min(mins):.1f}, max {max(mins):.1f}); 12 in parallel.", ""]
for name in ("prep", "train"):
    f = D / f"pyspy_{name}.txt"
    if not f.exists():
        continue
    total, per_kind, pids, acts, wait, func_incl = load(f)
    out += [f"## py-spy: {name} stages ({fmt_s(total)} process-time sampled at {RATE} Hz, idle included, main threads only)", "",
            "| process type | processes | process-time | share |", "|---|---|---|---|"]
    out += [f"| {k} | {len(pids[k])} | {fmt_s(v)} | {100 * v / total:.1f} % |" for k, v in per_kind.most_common()]
    if acts:
        tm = sum(acts.values()); np_ = max(len(pids["trainer: main process"]), 1)
        out += ["", f"### Training main process ({np_} runs): where the time goes", "",
                "| activity | per run | share | of which waiting on data / IPC |", "|---|---|---|---|"]
        out += [f"| {a} | {fmt_s(v / np_)} | {100 * v / tm:.1f} % | {100 * wait[a] / max(v, 1):.0f} % |" for a, v in acts.most_common()]
        out += ["", "Top functions in the training main process (inclusive time per run):", "", "| function | per run |", "|---|---|"]
        out += [f"| `{fn}` | {fmt_s(v / np_)} |" for fn, v in func_incl.most_common(40) if not fn.startswith(("<module>", "_bootstrap", "_find", "_call_with", "exec_module", "_load"))][:25]
    out.append("")
cpu = (D / "cpu_stat.log").read_text().split("\n") if (D / "cpu_stat.log").exists() else []
cpu = [l for l in cpu if "nr_periods" in l]
if len(cpu) > 1:
    g = lambda l, k: int(re.search(k + r" (\d+)", l).group(1))
    a, b = cpu[0], cpu[-1]; dt = int(b.split()[0]) - int(a.split()[0])
    out += ["## CPU (cgroup, 48-core quota)", "",
            f"Mean CPU use {(g(b, 'usage_usec') - g(a, 'usage_usec')) / 1e6 / max(dt, 1):.1f} cores; throttled in "
            f"{g(b, 'nr_throttled') - g(a, 'nr_throttled')} of {g(b, 'nr_periods') - g(a, 'nr_periods')} scheduler periods.", ""]
gl = [l.split() for l in (D / "gpu_dmon.log").read_text().splitlines() if l.strip() and not l.startswith("#")] if (D / "gpu_dmon.log").exists() else []
if gl:
    try:
        pw = [float(x[2]) for x in gl]; sm = [float(x[5]) for x in gl]; mem = [float(x[6]) for x in gl]
        out += ["## GPU (nvidia-smi dmon, every 10 s)", "",
                f"Power mean {sum(pw) / len(pw):.0f} W (cap 1000 W); SM activity mean {sum(sm) / len(sm):.0f} %; "
                f"memory-bandwidth activity mean {sum(mem) / len(mem):.0f} %; {len(gl)} samples.", ""]
    except (ValueError, IndexError):
        pass
out += ["Raw profiles: `pyspy_prep.txt`, `pyspy_train.txt` (collapsed stacks; open in https://www.speedscope.app or",
        "any flamegraph tool). Process-time = main-thread samples / 20 Hz per process (secondary threads excluded), so 12 parallel trainings add up to ~12x wall time."]
(D / "PROFILE_SUMMARY.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
