"""Build the markdown result tables used in README.md and docs/RL.md from results/*.csv."""
from pathlib import Path

import pandas as pd

from kaiju_cats import Board, library

R = Path(__file__).resolve().parents[1] / "results"


def pct(a, b):
    return f"{a / b:.0%}"


def real_table() -> str:
    boards = sorted((R.parent / "boards").glob("*.txt"))
    best = pd.Series({p.stem: library.lookup(Board.from_file(p))[1]["score"] for p in boards})
    search = pd.read_csv(R / "benchmark_careful.csv").groupby("board")["score"].agg(["mean", "max"])
    base = pd.read_csv(R / "baselines.csv").set_index("board") if (R / "baselines.csv").exists() else None
    rl0 = pd.read_csv(R / "rl_boards_120s_seed0.csv").set_index("board")
    extra = R / "rl_boards_120s_seeds12.csv"
    rl12 = pd.read_csv(extra) if extra.exists() else None
    rows = ["| Board | Best known | Random play (16k games) | Per-board CEM (20 s) | Policy sampling alone | **RL pipeline, 2 min** | RL pipeline, other 2 seeds | Search solver, 1 run (mean) | Search solver, best of 3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    tot = dict(best=0, rand=0, cem=0, samp=0, rl=0, s_mean=0, s_max=0)
    for b in best.index:
        o = ""
        if rl12 is not None:
            v = rl12[rl12.board == b]["final"].tolist()
            o = " / ".join(f"{x:,}" for x in v)
        rb = base.loc[b] if base is not None else None
        rows.append(f"| {b} | {best[b]:,} | {rb.random_16k_games:,} ({pct(rb.random_16k_games, best[b])}) | {rb.tabular_cem_20s:,} ({pct(rb.tabular_cem_20s, best[b])}) | "
                    f"{rl0.loc[b, 'sample']:,} ({pct(rl0.loc[b, 'sample'], best[b])}) | **{rl0.loc[b, 'final']:,} ({pct(rl0.loc[b, 'final'], best[b])})** | {o} | "
                    f"{search.loc[b, 'mean']:,.0f} ({pct(search.loc[b, 'mean'], best[b])}) | {search.loc[b, 'max']:,} ({pct(search.loc[b, 'max'], best[b])}) |"
                    if rb is not None else "")
        tot["best"] += best[b]; tot["samp"] += rl0.loc[b, "sample"]; tot["rl"] += rl0.loc[b, "final"]
        tot["s_mean"] += search.loc[b, "mean"]; tot["s_max"] += search.loc[b, "max"]
        if rb is not None:
            tot["rand"] += rb.random_16k_games; tot["cem"] += rb.tabular_cem_20s
    rows.append(f"| **All 8 (sum)** | {tot['best']:,} | {pct(tot['rand'], tot['best'])} | {pct(tot['cem'], tot['best'])} | {pct(tot['samp'], tot['best'])} | **{pct(tot['rl'], tot['best'])}** | | {pct(tot['s_mean'], tot['best'])} | {pct(tot['s_max'], tot['best'])} |")
    return "\n".join(rows)


def val_table() -> str:
    rl = pd.read_csv(R / "rl_boards_val_120s_seed0.csv").set_index("board")
    rows = ["| Validation board | Search solver, best of 2 runs (about 6 min) | Policy sampling alone | **RL pipeline, 2 min** |", "|---|---:|---:|---:|"]
    t_ref = t_s = t_rl = 0
    for b, r in rl.iterrows():
        rows.append(f"| {b} | {r.reference:,} | {r['sample']:,} ({pct(r['sample'], r.reference)}) | **{r.final:,} ({pct(r.final, r.reference)})** |")
        t_ref += r.reference; t_s += r["sample"]; t_rl += r.final
    rows.append(f"| **All 8 (sum)** | {t_ref:,} | {pct(t_s, t_ref)} | **{pct(t_rl, t_ref)}** |")
    return "\n".join(rows)


if __name__ == "__main__":
    print("### Real boards\n")
    print(real_table())
    print("\n### Held-out generated boards\n")
    print(val_table())
