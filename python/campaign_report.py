"""Figures from actual campaign progress and match summaries, including failures."""

import json
from pathlib import Path


def figures(run, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "figure.facecolor": "#fafaf7",
            "axes.facecolor": "#fafaf7",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    progress = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, kind in enumerate(("nnue", "cnn")):
        path = run / kind / "generation-0000/collection-progress.json"
        row = (
            json.loads(path.read_text())
            if path.exists()
            else {"completed": 0, "requested": 6250}
        )
        progress[kind] = row
        axes[0].barh(kind.upper(), row["completed"], color=("#267c80", "#d79537")[i])
        axes[0].text(
            row["completed"] + 80, i, f"{row['completed']:,}/6,250", va="center"
        )
        path = run / kind / f"generation-0000/{kind}.progress.json"
        if path.exists():
            metrics = json.loads(path.read_text())["metrics"]
            axes[1].plot(
                [m["epoch"] for m in metrics],
                [m["validation_loss"] for m in metrics],
                marker="o",
                label=kind.upper(),
            )
    axes[0].set_xlim(0, 7500)
    axes[0].set_xlabel(
        "Completed collection games; 5,000 train + 1,250 validation each"
    )
    axes[1].set_xlabel("Completed epoch")
    axes[1].set_ylabel("Validation loss (different objectives)")
    if axes[1].lines:
        axes[1].legend(frameon=False)
    fig.suptitle("Skallagrim league: actual progress", weight="bold")
    fig.tight_layout()
    fig.savefig(output / "progress.png", dpi=150)
    plt.close(fig)
    summaries = []
    for path in sorted(run.rglob("*-summary.json")):
        if "external" not in path.parts and not any(
            part.startswith("gate-") for part in path.parts
        ):
            continue
        value = json.loads(path.read_text())
        if value.get("games"):
            summaries.append(
                (str(path.relative_to(run)).replace("-summary.json", ""), value)
            )
    if summaries:
        fig, ax = plt.subplots(figsize=(12, max(5, len(summaries) * 0.48)))
        for i, (label, row) in enumerate(summaries):
            score = row.get("score_rate", 0) * 100
            interval = row.get("paired_bootstrap_95")
            ax.plot(score, i, "o", color="#267c80")
            if interval:
                ax.hlines(
                    i,
                    interval[0] * 100,
                    interval[1] * 100,
                    color="#267c80",
                    linewidth=2,
                )
            for offset, side, color in (
                (-0.16, "Attacker", "#d79537"),
                (0.16, "Defender", "#7267a5"),
            ):
                role = row.get("by_side", {}).get(side, {})
                if role.get("score_rate") is not None:
                    ax.plot(
                        role["score_rate"] * 100,
                        i + offset,
                        "s",
                        color=color,
                        markersize=5,
                    )
        labels = [
            f"{label} · {row['games']} games · {sum(row.get('termination_counts', {}).get(x, 0) for x in ('timeout', 'engine-failure', 'illegal-move', 'rules-incompatibility'))} failures"
            for label, row in summaries
        ]
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(-2, 102)
        ax.axvline(50, linestyle="--", color="#888")
        ax.set_xlabel(
            "Match score %; teal overall + paired 95% interval, gold attackers, purple defenders"
        )
        ax.set_title(
            "50-game / 300-ms gauntlets · scores include forfeits; inspect compatibility",
            weight="bold",
        )
        fig.tight_layout()
        fig.savefig(output / "gauntlets.png", dpi=150)
        plt.close(fig)
    return progress
