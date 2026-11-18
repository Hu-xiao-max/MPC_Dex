"""Plot existing independent physics audits; no new physics or learning."""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path(__file__).resolve().parent
SOURCE = D.parent / "mpc_clock_actor_recovery_0001"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    paths = {
        "policy": SOURCE / "inputs/clock_actor/independent_audit/physics.npz",
        "recovery": SOURCE / "recovery/independent_audit/physics.npz",
    }
    arrays = {}
    identities = {}
    for name, path in paths.items():
        copied = D / f"{name}_physics.npz"
        shutil.copyfile(path, copied)
        identities[name] = {"source": str(path), "sha256": sha(path)}
        assert sha(copied) == sha(path)
        with np.load(copied, allow_pickle=False) as data:
            columns = list(data["columns"])
            arrays[name] = {column: data["samples"][:, i] for i, column in enumerate(columns)}
    for key in arrays["policy"]:
        np.testing.assert_array_equal(arrays["policy"][key][:450], arrays["recovery"][key][:450])
    assert len(arrays["policy"]["time"]) == 950
    assert len(arrays["recovery"]["time"]) == 700
    assert arrays["policy"]["force_qualified_angle"].max() < 2 * np.pi
    assert arrays["recovery"]["force_qualified_angle"][-1] >= 2 * np.pi
    assert arrays["recovery"]["force_bearing_fingers"][-1] == 3

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, height_ratios=[2, 1])
    styles = {"policy": ("#526d82", "Clock BC policy alone"), "recovery": ("#087e6a", "Same prefix, then MPC")}
    for name, (color, label) in styles.items():
        data = arrays[name]
        times = data["time"] - data["time"][0] + .002
        axes[0].plot(times, np.rad2deg(data["force_qualified_angle"]), color=color, linewidth=1.8, label=label)
        axes[1].plot(times, np.rad2deg(data["axis_tilt_radians"]), color=color, linewidth=1.4)
    axes[0].axhline(360, color="#555555", linestyle="--", linewidth=1)
    axes[0].text(.03, 365, "Full-turn threshold", fontsize=9, color="#555555")
    axes[0].set_ylabel("Force-qualified rotation (degrees)")
    axes[0].set_ylim(-10, 435)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    axes[0].annotate("MPC-assisted success\n14 controls / 1.4 s", xy=(1.4, np.rad2deg(arrays["recovery"]["force_qualified_angle"][-1])), xytext=(1.43, 410), fontsize=9, color=styles["recovery"][0], arrowprops={"arrowstyle": "-", "color": styles["recovery"][0]})
    axes[0].annotate("Policy drops at 1.9 s", xy=(1.9, np.rad2deg(arrays["policy"]["force_qualified_angle"][-1])), xytext=(1.43, 170), fontsize=9, color=styles["policy"][0], arrowprops={"arrowstyle": "-", "color": styles["policy"][0]})
    axes[1].axhline(30, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Axis tilt (degrees)")
    axes[1].set_xlabel("Actual elapsed simulation time (seconds)")
    for ax in axes:
        ax.axvline(.9, color="#a35c00", linestyle=":", linewidth=1.2)
        ax.grid(axis="y", alpha=.16)
        ax.set_xlim(0, 1.97)
    axes[1].text(.92, 82, "MPC handoff", color="#a35c00", fontsize=9)
    fig.suptitle("MPC recovers an actual state visited by the clock policy", x=.09, ha="left", fontsize=14)
    fig.text(.09, .017, "Recorded 2 ms audits; first 0.9 s matches exactly. Diagnostic BC and MPC; no SAC policy success.\nRecovery stops at the first successful control boundary; no post-success hold was tested.", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, .08, 1, .95))
    fig.savefig(D / "comparison.png", dpi=180)
    fig.savefig(D / "comparison.pdf")
    plt.close(fig)
    report = {
        "purpose": "Visualization of previously recorded independent physics audits",
        "inputs": identities,
        "common_prefix_physics_rows_bitexact": 450,
        "new_physics_steps": 0,
        "new_optimizer_updates": 0,
        "source_policy_has_sac_updates": False,
        "mpc_assisted_recovery_is_policy_success": False,
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
    }
    (D / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (D / "manifest.json").write_text(json.dumps({p.name: sha(p) for p in sorted(D.iterdir()) if p.is_file() and p.name not in {"manifest.json", "run.log"}}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
