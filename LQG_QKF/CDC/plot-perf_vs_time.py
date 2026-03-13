import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np

from LQG_QKF import PUBLICATION_COLORS


BASE_DIR = os.path.dirname(__file__)
SIM_PKL_DIR = os.path.join(BASE_DIR, "sim_test", "pkl")
SIM_CACHE_DIR = os.path.join(BASE_DIR, "sim_test", "cache")

# Mapping from filter key used in tracking_sim to (pickle filename, pretty label)
FILTER_INFO = {
    "ekf": ("tracking_ekf.pkl", "LQG+EKF"),
    "ukf": ("tracking_ukf.pkl", "LQG+UKF"),
    "qkf_analytic": ("tracking_qkf_analytic.pkl", "LQG+QKF(analytic)"),
    "qkf_numeric": ("tracking_qkf_numeric.pkl", "LQG+QKF(numeric)"),
    "pf": ("tracking_pf.pkl", "LQG+PF"),
}


def load_results_from_cache():
    """
    Load per-trial tracking results for each filter from cache.

    Each tracking_trial=*.pkl file is the raw dict returned by run_one_trial_all_filters in tracking_sim:
      - for each filter key (ekf, ukf, ...):
          - tracking_error: 1D array over time
          - estimation_variance: 1D array over time
          - cost_to_go: 1D array over time (cost_to_go[0] = total LQG cost for that trial)
          - time_per_step: 1D array over time; total_time is also stored
    """
    data_points = []

    if not os.path.isdir(SIM_CACHE_DIR):
        print(f"[warning] Cache directory not found: {SIM_CACHE_DIR}")
        return data_points

    for fname in os.listdir(SIM_CACHE_DIR):
        if not fname.startswith("tracking_trial=") or not fname.endswith(".pkl"):
            continue
        path = os.path.join(SIM_CACHE_DIR, fname)
        with open(path, "rb") as f:
            trial_out = pkl.load(f)

        for key, (_, label) in FILTER_INFO.items():
            if key not in trial_out:
                continue
            res = trial_out[key]
            # Total time for this trial & filter
            total_time = float(res.get("total_time", np.sum(res.get("time_per_step", 0.0))))
            # Mean tracking error for this trial & filter
            te = np.asarray(res.get("tracking_error", []), dtype=float).ravel()
            mean_err = float(np.mean(te)) if te.size > 0 else np.nan
            # Total cost: cost_to_go[0]
            c2g = np.asarray(res.get("cost_to_go", []), dtype=float).ravel()
            total_cost = float(c2g[0]) if c2g.size > 0 else np.nan
            # Mean estimation variance
            ev = np.asarray(res.get("estimation_variance", []), dtype=float).ravel()
            mean_var = float(np.mean(ev)) if ev.size > 0 else np.nan

            data_points.append(
                {
                    "filter_key": key,
                    "label": label,
                    "color": PUBLICATION_COLORS.get(key, "gray"),
                    "total_time": total_time,
                    "mean_error": mean_err,
                    "total_cost": total_cost,
                    "mean_variance": mean_var,
                }
            )

    return data_points


def make_accuracy_vs_time_plot(save_path=None, show=False, metric: str = "cost"):
    """
    Scatter plot of a chosen performance metric vs. computation time for all filters.

    Using per-trial cache: each point is one (trial, filter) pair.

    x-axis (always): total_time  (seconds for that trial & filter)
    y-axis (metric) — all mapped so **higher is better**:
        - 'cost'     : 1 / total LQG cost per trial
        - 'accuracy' : 1 / mean tracking error per trial
        - 'variance' : 1 / mean estimation variance per trial
    """
    results = load_results_from_cache()
    if not results:
        print("No tracking cache results found; run tracking_sim.py first.")
        return

    metric = metric.lower()
    if metric not in ("cost", "accuracy", "variance"):
        raise ValueError("metric must be one of: 'cost', 'accuracy', 'variance'")

    # Wider figure to leave room for legend and improve readability
    fig, ax = plt.subplots(figsize=(6.0, 3.2))

    times = []
    values = []

    for pt in results:
        time_val = pt["total_time"]
        if metric == "cost":
            cost = pt["total_cost"]
            if cost <= 0:
                continue
            y_val = 1.0 / cost
        elif metric == "accuracy":
            err = pt["mean_error"]
            if err <= 0:
                continue
            y_val = 1.0 / err
        else:  # 'variance'
            var = pt["mean_variance"]
            if var <= 0:
                continue
            y_val = 1.0 / var
        times.append(time_val)
        values.append(y_val)
        ax.scatter(
            time_val,
            y_val,
            color=pt["color"],
            s=8,
            edgecolor="k",
            linewidth=0.5,
            label=pt["label"],
            alpha=0.85,
        )

    # Keep legend entries unique and place it outside the main axes
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # unique by label
    ax.legend(by_label.values(), by_label.keys(),
              fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1),
              frameon=True)

    ax.set_xlabel("Computation time (s, log$_2$ scale)")
    if metric == "cost":
        ax.set_ylabel("Performance (1 / total LQG cost)")
    elif metric == "accuracy":
        ax.set_ylabel("Performance (1 / mean tracking error)")
    else:  # variance
        ax.set_ylabel("Performance (1 / mean estimation variance)")
    ax.grid(True, alpha=0.3, which="both")

    # Log scales on both axes to spread clustered values
    try:
        ax.set_xscale("log", base=2)
    except TypeError:
        # Older Matplotlib versions
        ax.set_xscale("log", basex=2)
    ax.set_yscale("log")

    # # Optional reference lines similar to the sketch (median time / metric)
    # if times and values:
    #     x_ref = np.median(times)
    #     y_ref = np.median(values)
    #     ax.axvline(x_ref, color="red", linewidth=1.0)
    #     ax.axhline(y_ref, color="red", linewidth=1.0)

    plt.tight_layout(rect=[0, 0, 0.76, 1])

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "sim_test", "perf", f"{metric}_vs_time.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot performance metric vs computation time.")
    parser.add_argument(
        "--metric",
        type=str,
        default="cost",
        choices=["cost", "accuracy", "variance"],
        help=(
            "Y-axis metric: "
            "'cost' (total LQG cost), "
            "'accuracy' (1 / mean tracking error), or "
            "'variance' (1 / mean estimation variance). Default: cost."
        ),
    )
    args = parser.parse_args()
    make_accuracy_vs_time_plot(metric=args.metric)

