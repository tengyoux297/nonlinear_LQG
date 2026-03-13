# Tracking simulation: LQG with multiple filters (EKF, UKF, QKF, PF) following a Figure-8 reference.
# Uses LQG system from LQG_QKF.py. Outputs: sim_test/pkl/, sim_test/perf/, sim_test/cache/.

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from stateDynamics import StateDynamics, sensor, Vec
from LQG_QKF import LQG, PUBLICATION_COLORS

# ---------------------------------------------------------------------------
# Directories (same layout as prelim_test)
# ---------------------------------------------------------------------------
sim_test_dir = 'sim_test/'
sim_pkl_dir = sim_test_dir + 'pkl/'
sim_perf_dir = sim_test_dir + 'perf/'
sim_cache_dir = sim_test_dir + 'cache/'
for d in (sim_pkl_dir, sim_perf_dir, sim_cache_dir):
    os.makedirs(d, exist_ok=True)

# Filter configs: (filter_type, lqr_type) for LQG. QKF uses aug_analytic / aug_numeric.
FILTER_CONFIGS = [
    ('ekf', 'orig'),
    ('ukf', 'orig'),
    ('qkf', 'aug_analytic'),
    ('qkf', 'aug_numeric'),
    ('pf', 'orig'),
]
FILTER_KEYS = ['ekf', 'ukf', 'qkf_analytic', 'qkf_numeric', 'pf']
FILTER_LABELS = {
    'ekf': 'LQG+EKF',
    'ukf': 'LQG+UKF',
    'qkf_analytic': 'LQG+QKF(analytic)',
    'qkf_numeric': 'LQG+QKF(numeric)',
    'pf': 'LQG+PF',
}

# ---------------------------------------------------------------------------
# 1. Double-integrator dynamics: x = [px, vx, py, vy], u = [ax, ay]
# ---------------------------------------------------------------------------
dt = 0.1
n1, n2, p = 2, 2, 2
n = n1 + n2  # 4

A_E = np.array([[1.0, dt], [0.0, 1.0]])
A_S = np.array([[1.0, dt], [0.0, 1.0]])
B_E = np.array([[0.5 * dt**2, 0.0], [dt, 0.0]])
B_S = np.array([[0.0, 0.5 * dt**2], [0.0, dt]])

# Process noise: fixed scaling for consistent filter behavior (small => plant easier to control)
NOISE_SCALE_PROCESS = 0.1   # <1 reduces effective process noise for tighter tracking
w_scale_base = 1e-3
w_scale = w_scale_base * NOISE_SCALE_PROCESS
# Scale by state "size": position variance ~ 1, velocity variance ~ (1/dt)^2 relative to position
W_diag = np.array([1.0, 0.5 / dt, 1.0, 0.5 / dt])  # position, velocity (x,y)
W = w_scale * np.diag(W_diag)

# ---------------------------------------------------------------------------
# 2. Lemniscate of Bernoulli reference (Figure-8)
# ---------------------------------------------------------------------------
def lemniscate_position(t, a=2.0):
    s = np.sin(t)
    c = np.cos(t)
    denom = 1.0 + s**2
    px = a * c / denom
    py = a * s * c / denom
    return px, py

def reference_trajectory(H, dt, a=2.0):
    """Reference states [px, vx, py, vy] at steps 0..H (length H+1)."""
    ref = []
    for k in range(H + 1):
        t = k * dt
        px, py = lemniscate_position(t, a)
        px1, py1 = lemniscate_position(t + dt, a)
        vx = (px1 - px) / dt
        vy = (py1 - py) / dt
        ref.append(np.array([[px], [vx], [py], [vy]]))
    return ref

# ---------------------------------------------------------------------------
# 3. Quadratic observation: 20 fixed sensors
# ---------------------------------------------------------------------------
m = 20
np.random.seed(42)
sensor_xy = np.random.uniform(-2.5, 2.5, (m, 2))

C = np.zeros((m, n))
M = np.zeros((m, n, n))
measA = np.zeros((m, 1))
for i in range(m):
    sx, sy = sensor_xy[i, 0], sensor_xy[i, 1]
    C[i, :] = [-2 * sx, 0.0, -2 * sy, 0.0]
    M[i, 0, 0] = 1.0
    M[i, 2, 2] = 1.0
    measA[i, 0] = sx**2 + sy**2

# Measurement noise: fixed scaling (moderate so filters don't overfit)
NOISE_SCALE_MEAS = 1.0
v_scale = 1e-1 * NOISE_SCALE_MEAS
V = (v_scale ** 2) * np.eye(m)

# ---------------------------------------------------------------------------
# 4. LQR cost
# ---------------------------------------------------------------------------
# Strong state penalty and lower R for aggressive tracking
Q_small = np.diag([25.0, 2.0, 25.0, 2.0])
Q = np.zeros((n + n**2, n + n**2))
Q[:n, :n] = Q_small
R = 0.12 * np.eye(p)

# ---------------------------------------------------------------------------
# 5. One trial: same setting, same reference path; each filter has its own actual trajectory
# ---------------------------------------------------------------------------
def run_one_trial_all_filters(H, ref_traj, x0, goal0, rand_seed=0, n_particles=500):
    """
    Run one trial with the same reference path but independent actual trajectories per filter.
    Each filter's own dynamics F is advanced using its control.
    Returns dict with per-filter result dicts compatible with aggregate_trials.
    """
    np.random.seed(rand_seed)
    # Single shared sensor model; measurements depend on each filter's actual state x_actual
    shared_sensor = sensor(C, M, V, measA=measA)

    # Five LQG instances (same setting, different filter)
    lqgs = []
    for idx, filter_key in enumerate(FILTER_KEYS):
        ft, lt = FILTER_CONFIGS[idx][0], FILTER_CONFIGS[idx][1]
        kw = dict(n1=n1, n2=n2, p=p, W=W, A_E=A_E, A_S=A_S, B_S=B_S, C=C, M=M, V=V, Q=Q, R=R,
                  goal_state=goal0, H=H, filter_type=ft, lqr_type=lt, B_E=B_E, measA=measA)
        if ft == 'pf':
            kw['n_particles'] = n_particles
        lqg = LQG(**kw)
        lqg.F.set_x(x0.copy())  # actual state for this filter
        lqg.x_hat = x0.copy()
        lqg.sync_qkf_initial_state()  # so QKF Z_est matches x0 and path does not start with a jump
        lqgs.append((filter_key, lqg))

    ref_path = [ref_traj[0].copy()]
    per_filter = {
        fk: {
            'actual_path': [lqg.F.get_x().copy()],
            'est_path': [lqg.x_hat.copy()],
            'tracking_error': [],
            'control_effort': [],
            'stage_cost': [],
            'estimation_var': [],
            'mse_actual_goal': [],
            'mse_est_goal': [],
            'time_per_step': [],
        }
        for fk, lqg in lqgs
    }

    # Reference preview: steer toward ref a few steps ahead so we lead the curve (smoother tracking)
    REF_PREVIEW_STEPS = 3
    for step in range(1, H + 1):
        # Goal = reference at end of step, with optional preview (ahead) for feedforward
        ref_idx = min(step + REF_PREVIEW_STEPS, len(ref_traj) - 1)
        x_ref = ref_traj[ref_idx]
        for fk, lqg in lqgs:
            t0 = time.perf_counter()
            # For each filter: measure its own actual state, update estimator and controller,
            # then advance its dynamics.
            x_actual = lqg.F.get_x()
            y_meas = shared_sensor.measure(x_actual)
            lqg.set_goal_state(x_ref)
            lqg.update_lqe(y_meas=y_meas)
            lqg.update_lqr()
            lqg.F.forward()
            per_filter[fk]['time_per_step'].append(time.perf_counter() - t0)
        x_ref_step = ref_traj[min(step, len(ref_traj) - 1)]
        ref_path.append(x_ref_step.copy())
        for fk, lqg in lqgs:
            x_actual = lqg.F.get_x()
            x_est = lqg.x_hat
            u = lqg.F.get_u()
            per_filter[fk]['actual_path'].append(x_actual.copy())
            per_filter[fk]['est_path'].append(x_est.copy())
            per_filter[fk]['tracking_error'].append(np.linalg.norm(x_actual[:2] - x_ref_step[:2]).item())
            per_filter[fk]['control_effort'].append(np.linalg.norm(u).item())
            dx = (x_est - x_ref_step).reshape(-1, 1)
            per_filter[fk]['stage_cost'].append((dx.T @ Q_small @ dx + u.T @ R @ u).item())
            per_filter[fk]['mse_actual_goal'].append(np.sum((x_actual - x_ref_step) ** 2).item())
            per_filter[fk]['mse_est_goal'].append(np.sum((x_est - x_ref_step) ** 2).item())
            var_t = np.trace(lqg.Pz_est[:lqg.n, :lqg.n]) if lqg.filter_type == 'qkf' else np.trace(lqg.P_est)
            per_filter[fk]['estimation_var'].append(np.asarray(var_t).item())

    # Build return: per-filter dicts compatible with aggregate_trials
    out = {'ref_path': np.array(ref_path).squeeze()}
    for fk, _ in lqgs:
        sc = np.array(per_filter[fk]['stage_cost'])
        ap = np.array(per_filter[fk]['actual_path']).squeeze()
        out[fk] = {
            'true_path': ap,  # keep key name for compatibility; values are actual trajectories
            'est_path': np.array(per_filter[fk]['est_path']).squeeze(),
            'ref_path': out['ref_path'],
            'tracking_error': np.array(per_filter[fk]['tracking_error']),
            'control_effort': np.array(per_filter[fk]['control_effort']),
            'stage_cost': sc,
            'cost_to_go': np.array([np.sum(sc[k:]) for k in range(len(sc))]),
            'estimation_variance': np.array(per_filter[fk]['estimation_var']),
            'mse_actual_goal': np.array(per_filter[fk]['mse_actual_goal']),
            'mse_est_goal': np.array(per_filter[fk]['mse_est_goal']),
            'time_per_step': np.array(per_filter[fk]['time_per_step']),
            'total_time': np.sum(per_filter[fk]['time_per_step']),
            'filter_key': fk,
        }
    return out

# ---------------------------------------------------------------------------
# 6. Aggregate trials: mean and std over trials for each filter
# ---------------------------------------------------------------------------
def aggregate_trials(trial_results_list):
    """trial_results_list: list of dicts from run_tracking_one_filter. Returns dict with mean_*, std_*."""
    if not trial_results_list:
        return None
    n_trials = len(trial_results_list)
    r0 = trial_results_list[0]
    true_path = np.array([r['true_path'] for r in trial_results_list])
    est_path = np.array([r['est_path'] for r in trial_results_list])
    ref_path = np.array([r['ref_path'] for r in trial_results_list])
    tracking_error = np.array([r['tracking_error'] for r in trial_results_list])
    control_effort = np.array([r['control_effort'] for r in trial_results_list])
    per_trial_mean_err = np.mean(tracking_error, axis=1)
    per_trial_mean_effort = np.mean(control_effort, axis=1)

    out = {
        'true_path_mean': np.mean(true_path, axis=0),
        'est_path_mean': np.mean(est_path, axis=0),
        'est_path_std': np.std(est_path, axis=0),
        'ref_path_mean': np.mean(ref_path, axis=0),
        'tracking_error_mean': np.mean(tracking_error, axis=0),
        'tracking_error_std': np.std(tracking_error, axis=0),
        'control_effort_mean': np.mean(control_effort, axis=0),
        'control_effort_std': np.std(control_effort, axis=0),
        'n_trials': n_trials,
        'mean_tracking_error': np.mean(per_trial_mean_err),
        'std_tracking_error': np.std(per_trial_mean_err),
        'mean_control_effort': np.mean(per_trial_mean_effort),
        'std_control_effort': np.std(per_trial_mean_effort),
    }
    if 'cost_to_go' in r0:
        c2g = np.array([r['cost_to_go'] for r in trial_results_list])
        out['cost_to_go_mean'] = np.mean(c2g, axis=0)
        out['cost_to_go_std'] = np.std(c2g, axis=0)
    if 'estimation_variance' in r0:
        ev = np.array([r['estimation_variance'] for r in trial_results_list])
        out['estimation_variance_mean'] = np.mean(ev, axis=0)
        out['estimation_variance_std'] = np.std(ev, axis=0)
    if 'mse_actual_goal' in r0:
        mse = np.array([r['mse_actual_goal'] for r in trial_results_list])
        out['mse_actual_goal_mean'] = np.mean(mse, axis=0)
        out['mse_actual_goal_std'] = np.std(mse, axis=0)
    if 'mse_est_goal' in r0:
        mse_est = np.array([r['mse_est_goal'] for r in trial_results_list])
        out['mse_est_goal_mean'] = np.mean(mse_est, axis=0)
        out['mse_est_goal_std'] = np.std(mse_est, axis=0)
    if 'time_per_step' in r0:
        tps = np.array([r['time_per_step'] for r in trial_results_list])
        out['time_per_step_mean'] = np.mean(tps, axis=0)
        out['time_per_step_std'] = np.std(tps, axis=0)
        out['total_time_mean'] = np.mean([r['total_time'] for r in trial_results_list])
        out['total_time_std'] = np.std([r['total_time'] for r in trial_results_list])
    return out

# ---------------------------------------------------------------------------
# 7. Main: multi-trial run with cache resume, aggregate, save pkl, plot
# ---------------------------------------------------------------------------
H = 200
trials = 100
rand_seed_base = 64
ref_traj = reference_trajectory(H, dt)
x0 = ref_traj[0].copy()
goal0 = ref_traj[0].copy()

# One trial = one shared true path, same reference; all filters see same measurements. Cache per trial.
all_trials_by_filter = {k: [] for k in FILTER_KEYS}
for i in tqdm(range(trials), desc='Trials', leave=True, position=0):
    cache_path = os.path.join(sim_cache_dir, f'tracking_trial={i}.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            trial_out = pkl.load(f)
    else:
        trial_out = run_one_trial_all_filters(H, ref_traj, x0, goal0, rand_seed=rand_seed_base + i)
        with open(cache_path, 'wb') as f:
            pkl.dump(trial_out, f)
    for fk in FILTER_KEYS:
        all_trials_by_filter[fk].append(trial_out[fk])

# Aggregate per filter and save to pkl
results = {}
for filter_key in FILTER_KEYS:
    agg = aggregate_trials(all_trials_by_filter[filter_key])
    results[filter_key] = agg
    pkl_path = os.path.join(sim_pkl_dir, f'tracking_{filter_key}.pkl')
    with open(pkl_path, 'wb') as f:
        pkl.dump(agg, f)

# Paths plot: reference + each system's path. Use one randomly selected trial (no mean actual).
ref_path_mean = results[FILTER_KEYS[0]]['ref_path_mean']
trial_idx = np.random.randint(0, trials)

# ---------------------------------------------------------------------------
# 8. Final results and comparison plots
# ---------------------------------------------------------------------------
# (1) Paths: reference + each filter's path from one trial (no mean actual). View area around reference.
fig, ax = plt.subplots(figsize=(8, 7))
# Line styles: QKF analytic dashed so it stays visible when overlapping QKF numeric
# Distinct line style per path so overlapping trajectories stay distinguishable
PATH_LINESTYLE = {
    'ekf': '-',           # solid
    'ukf': '--',          # dashed
    'qkf_analytic': '-.', # dash-dot
    'qkf_numeric': ':',   # dotted
    'pf': (0, (3, 1, 1, 1)),  # dash-dot-dot
}
ax.plot(ref_path_mean[:, 0], ref_path_mean[:, 2], 'k-', lw=2, label='Reference (Figure-8)', zorder=5)
for filter_key in FILTER_KEYS:
    est = all_trials_by_filter[filter_key][trial_idx]['est_path']
    c = PUBLICATION_COLORS.get(filter_key, 'gray')
    ls = PATH_LINESTYLE.get(filter_key, '-')
    ax.plot(est[:, 0], est[:, 2], color=c, lw=1.5, ls=ls, alpha=0.75, label=FILTER_LABELS[filter_key])
ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], c='gray', s=20, alpha=0.7, label='Sensors', zorder=3)
# Scope view strictly to the reference path bounding box + small margin (so plot is clearly zoomed)
px_ref = ref_path_mean[:, 0]
py_ref = ref_path_mean[:, 2]
x_min_r, x_max_r = px_ref.min(), px_ref.max()
y_min_r, y_max_r = py_ref.min(), py_ref.max()
span_x = x_max_r - x_min_r
span_y = y_max_r - y_min_r
margin = 0.15 * max(span_x, span_y, 0.2)  # 15% padding around reference
ax.set_xlim(x_min_r - margin, x_max_r + margin)
ax.set_ylim(y_min_r - margin, y_max_r + margin)
# Equal aspect without letting matplotlib expand limits (keep our zoom)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('$p_x$')
ax.set_ylabel('$p_y$')
ax.set_title(f'Paths: Reference and each system (one trial, trial={trial_idx})')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True)
ax.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space for legend on the right
plt.savefig(os.path.join(sim_perf_dir, 'tracking_paths_actual_vs_reference.png'), dpi=150, bbox_inches='tight')
plt.close()

# (2) Estimation variance (trace P) and MSE (estimate vs reference) vs time
# Line styles so overlapping curves (e.g. QKF analytic vs numeric) stay distinguishable
VAR_MSE_LINESTYLES = {'ekf': '-', 'ukf': '--', 'qkf_analytic': '-.', 'qkf_numeric': ':', 'pf': (0, (3, 1, 1, 1))}
if 'estimation_variance_mean' in results[FILTER_KEYS[0]] and 'mse_est_goal_mean' in results[FILTER_KEYS[0]]:
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    for filter_key in FILTER_KEYS:
        t = np.arange(len(results[filter_key]['estimation_variance_mean']))
        mu_v = results[filter_key]['estimation_variance_mean']
        std_v = results[filter_key].get('estimation_variance_std', np.zeros_like(mu_v))
        mu_v_plot = np.maximum(mu_v, 1e-12)
        lo_v = np.maximum(mu_v - std_v, 1e-12)
        hi_v = np.maximum(mu_v + std_v, 1e-12)
        ls = VAR_MSE_LINESTYLES.get(filter_key, '-')
        axes[0].plot(mu_v_plot, color=PUBLICATION_COLORS.get(filter_key, 'gray'), lw=1.5, ls=ls, label=FILTER_LABELS[filter_key])
        axes[0].fill_between(t, lo_v, hi_v, color=PUBLICATION_COLORS.get(filter_key, 'gray'), alpha=0.2)
    axes[0].set_ylabel('Estimation variance (trace P)')
    axes[0].set_yscale('log')
    axes[0].set_title(f'Estimation Variance vs Time (n_trials={trials})')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True)
    axes[0].grid(True, alpha=0.3, which='both')
    for filter_key in FILTER_KEYS:
        t = np.arange(len(results[filter_key]['mse_est_goal_mean']))
        mu_m = results[filter_key]['mse_est_goal_mean']
        std_m = results[filter_key].get('mse_est_goal_std', np.zeros_like(mu_m))
        mu_plot = np.maximum(mu_m, 1e-10)
        lo = np.maximum(mu_m - std_m, 1e-10)
        hi = np.maximum(mu_m + std_m, 1e-10)
        ls = VAR_MSE_LINESTYLES.get(filter_key, '-')
        axes[1].plot(mu_plot, color=PUBLICATION_COLORS.get(filter_key, 'gray'), lw=1.5, ls=ls, label=FILTER_LABELS[filter_key])
        axes[1].fill_between(t, lo, hi, color=PUBLICATION_COLORS.get(filter_key, 'gray'), alpha=0.2)
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('MSE (estimate vs reference)')
    axes[1].set_yscale('log')
    axes[1].set_title('MSE (Estimate vs Reference) vs Time')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True)
    axes[1].grid(True, alpha=0.3, which='both')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(sim_perf_dir, 'tracking_estimation_variance_and_mse.png'), dpi=150, bbox_inches='tight')
# (3) Control effort (mean ± std) and running time (mean ± std total time per filter)
if 'total_time_mean' in results[FILTER_KEYS[0]]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [FILTER_LABELS[k] for k in FILTER_KEYS]
    colors = [PUBLICATION_COLORS.get(k, 'gray') for k in FILTER_KEYS]

    # Control effort: mean ||u|| and std over trials
    mean_effort = [results[k]['mean_control_effort'] for k in FILTER_KEYS]
    std_effort = [results[k].get('std_control_effort', 0) for k in FILTER_KEYS]
    axes[0].bar(labels, mean_effort, yerr=std_effort, color=colors, edgecolor='black', capsize=5)
    axes[0].set_ylabel('Mean control effort ||u||')
    axes[0].set_title(f'Control Effort by Filter (mean ± std, n_trials={trials})')
    axes[0].tick_params(axis='x', rotation=15)

    # Running time: mean ± std total time per filter
    total_mean = [results[k]['total_time_mean'] for k in FILTER_KEYS]
    total_std = [results[k].get('total_time_std', 0) for k in FILTER_KEYS]
    axes[1].bar(labels, total_mean, yerr=total_std, color=colors, edgecolor='black', capsize=5)
    axes[1].set_ylabel('Total running time (s)')
    axes[1].set_title(f'Running Time per Trial (mean ± std, n_trials={trials})')
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(sim_perf_dir, 'tracking_control_effort_and_time.png'), dpi=150, bbox_inches='tight')
    plt.close()

print('Tracking sim done. Results in:', sim_test_dir)
print('  trials:', trials)
print('  pkl (aggregated): ', sim_pkl_dir)
print('  cache (per-trial):', sim_cache_dir)
print('  plots:             ', sim_perf_dir)

# plot performance vs time
import subprocess
subprocess.run(['python', 'plot-perf_vs_time.py', '--metric', 'cost'])
subprocess.run(['python', 'plot-perf_vs_time.py', '--metric', 'accuracy'])
subprocess.run(['python', 'plot-perf_vs_time.py', '--metric', 'variance'])