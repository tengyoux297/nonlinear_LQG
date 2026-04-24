import os
import pickle as pkl
import numpy as np

base = r"D:\AC\UCLA\ECE\UCLA_LEMUR\nonlinear_LQG\LQG_QKF\CDC"

print("SIM TEST")
sim = os.path.join(base, "sim_test", "pkl")
for k in ["ekf", "ukf", "qkf_analytic", "qkf_numeric", "pf", "ckf"]:
    with open(os.path.join(sim, f"tracking_{k}.pkl"), "rb") as f:
        d = pkl.load(f)
    mse = np.array(d.get("mse_est_goal_mean", []), dtype=float)
    te = np.array(d.get("tracking_error_mean", []), dtype=float)
    print(
        k,
        "mean_mse_est_goal=", float(np.mean(mse)) if mse.size else None,
        "final_mse_est_goal=", float(mse[-1]) if mse.size else None,
        "mean_tracking_err=", float(np.mean(te)) if te.size else None,
    )

print("\nSIM TEST component breakdown from cache (est_path vs ref_path)")
cache = os.path.join(base, "sim_test", "cache")
for k in ["ekf", "ukf", "qkf_analytic", "qkf_numeric", "pf", "ckf"]:
    pos_mse = []
    vel_mse = []
    total_mse = []
    for fn in os.listdir(cache):
        if not (fn.startswith("tracking_trial=") and fn.endswith(".pkl")):
            continue
        with open(os.path.join(cache, fn), "rb") as f:
            tr = pkl.load(f)
        e = np.asarray(tr[k]["est_path"], dtype=float)
        r = np.asarray(tr[k]["ref_path"], dtype=float)
        d = e - r
        pos_mse.append(np.mean(d[:, [0, 2]] ** 2))
        vel_mse.append(np.mean(d[:, [1, 3]] ** 2))
        total_mse.append(np.mean(np.sum(d**2, axis=1)))
    print(
        k,
        "pos_mse=", float(np.mean(pos_mse)),
        "vel_mse=", float(np.mean(vel_mse)),
        "total_mse=", float(np.mean(total_mse)),
    )

print("\nPRELIM TEST (mscale=100)")
pre = os.path.join(base, "prelim_test", "pkl")
mp = {
    "ekf": "ekf_results-mscale=100.pkl",
    "ukf": "ukf_results-mscale=100.pkl",
    "qkf_analytic": "qkf_analytic_results-mscale=100.pkl",
    "qkf_numeric": "qkf_results-mscale=100.pkl",
    "pf": "pf_results-mscale=100.pkl",
    "ckf": "ckf_results-mscale=100.pkl",
}
for k, fn in mp.items():
    with open(os.path.join(pre, fn), "rb") as f:
        d = pkl.load(f)
    err = np.array(d[0], dtype=float)
    print(
        k,
        "mean_err=", float(np.mean(err)),
        "final_err=", float(np.mean(err[:, -1])) if err.ndim == 2 else None,
    )