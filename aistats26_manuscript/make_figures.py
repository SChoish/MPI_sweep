from pathlib import Path
import argparse
import csv
import statistics
from collections import defaultdict
import numpy as np
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 9.0
mpl.rcParams['axes.prop_cycle'] = cycler(color=['0.05', '0.25', '0.45', '0.65', '0.80'])
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Build manuscript figures from released CSV results.')
parser.add_argument('--results-dir', type=Path, required=True)
args = parser.parse_args()
RESULTS = args.results_dir.resolve()
if not RESULTS.is_dir():
    raise FileNotFoundError(f'results directory not found: {RESULTS}')

OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


def read_rows(relative, required=()):
    path = RESULTS / relative
    if not path.is_file():
        raise FileNotFoundError(f'required result file not found: {path}')
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = set(required) - fields
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: no data rows')
    return rows


def run_key(row):
    return row['env'], float(row['tau']), int(row['seed'])


def load_sweep(hops, integrator, seeds):
    values = {}
    environments = None
    for seed in seeds:
        rows = read_rows(Path(hops) / integrator / f'seed{seed}.csv', ('tau',))
        current = [field for field in rows[0] if field != 'tau']
        if environments is None:
            environments = current
        elif current != environments:
            raise ValueError(f'{hops}/{integrator}: environment columns differ across seeds')
        for row in rows:
            total_budget = float(row['tau'])
            for env in environments:
                value = row[env].strip()
                if value and value != '—':
                    key = total_budget, seed, env
                    if key in values:
                        raise ValueError(f'{hops}/{integrator}: duplicate cell {key}')
                    values[key] = float(value)
    return environments, values


method_specs = [
    ('TD3+BC', 'K=1', 'Imp', (0, 1, 2, 3), 'o'),
    ('PART-Prox (K=2)', 'K=2', 'Imp', (0, 1, 2, 3), 's'),
    ('PART-Prox (K=3)', 'K=3', 'Imp', (0, 1, 2, 3), '^'),
    ('PART-Prox (K=4)', 'K=4', 'Imp', (0, 1, 2, 3), 'P'),
    ('PART-Lin (K=2)', 'K=2', 'Exp', (0, 1, 2, 3), 'D'),
    ('PART-Lin (K=3)', 'K=3', 'Exp', (0, 1, 2, 3), 'v'),
]
sweep = []
common_taus = None
reference_envs = None
for label, hops, integrator, seeds, marker in method_specs:
    envs, values = load_sweep(hops, integrator, seeds)
    if reference_envs is None:
        reference_envs = envs
    elif envs != reference_envs:
        raise ValueError(f'{label}: environment columns differ from the other methods')
    method_taus = {key[0] for key in values}
    common_taus = method_taus if common_taus is None else common_taus & method_taus
    sweep.append((label, marker, values, seeds))
tau = np.array(sorted(common_taus), dtype=float)
if not len(tau):
    raise ValueError('the released sweep files have no common budgets')
curves = []
for label, marker, values, seeds in sweep:
    curve = np.array([
        statistics.mean(values[(total_budget, seed, env)] for seed in seeds for env in reference_envs)
        for total_budget in tau
    ])
    curves.append((curve, label, marker))

fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.30), constrained_layout=True)
ax = axs[0]
for y, lab, marker in curves:
    ax.plot(tau, y, marker=marker, markersize=3.2, linewidth=1.25, label=lab)
ax.set_xscale('log')
ax.set_xlabel('total nominal budget $T$')
ax.set_ylabel('mean D4RL score')
ax.set_title('(a) Stability envelope')
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.legend(frameon=False, fontsize=7, ncol=2, loc='best')

methods = [label.replace('PART-', '').replace(' (K=', '-').replace(')', '') for _, label, _ in curves]
high_taus = [total_budget for total_budget in tau if total_budget >= 4]
n_cells = len(high_taus) * len(reference_envs)
seedmean = []
raw = []
run_denominators = []
for _, _, values, seeds in sweep:
    cell_scores = [
        statistics.mean(values[(total_budget, seed, env)] for seed in seeds)
        for total_budget in high_taus for env in reference_envs
    ]
    run_scores = [
        values[(total_budget, seed, env)]
        for total_budget in high_taus for seed in seeds for env in reference_envs
    ]
    seedmean.append(sum(score < 20 for score in cell_scores))
    raw.append(sum(score < 20 for score in run_scores))
    run_denominators.append(len(run_scores))
seedmean = np.asarray(seedmean)
raw = np.asarray(raw)
seedmean_rate = seedmean / n_cells
raw_rate = raw / np.asarray(run_denominators)
x = np.arange(len(methods))
width = 0.36
axs[1].bar(x-width/2, seedmean_rate, width, label='environment--budget cells')
axs[1].bar(x+width/2, raw_rate, width, label='seeded runs')
axs[1].set_xticks(x, methods, rotation=18, ha='right')
axs[1].set_ylabel('collapse rate')
axs[1].set_ylim(0,0.65)
axs[1].set_title(r'(b) High-budget collapse ($T\geq4$)')
axs[1].grid(True, axis='y', alpha=0.25, linewidth=0.5)
axs[1].legend(frameon=False, fontsize=7, loc='upper right')
for i,(a,b,ar,br,n_runs) in enumerate(zip(seedmean,raw,seedmean_rate,raw_rate,run_denominators)):
    axs[1].text(i-width/2, ar+0.012, f'{a}/{n_cells}', ha='center', va='bottom', fontsize=6.3)
    axs[1].text(i+width/2, br+0.012, f'{b}/{n_runs}', ha='center', va='bottom', fontsize=6.3)
fig.savefig(OUT/'stability_envelope.pdf', bbox_inches='tight')
fig.savefig(OUT/'stability_envelope.png', dpi=240, bbox_inches='tight')
plt.close(fig)

# Environment-wise Prox-2 response from the released two-seed diagnostic summary.
env_rows = read_rows(
    Path('diagnostics/environment_t_sensitivity.csv'),
    ('env', 'tau', 'd4rl_score_mean', 'dcrit_mean'),
)
env_labels = [
    ('halfcheetah-medium-v2', 'HalfCheetah-M'),
    ('halfcheetah-medium-replay-v2', 'HalfCheetah-MR'),
    ('halfcheetah-expert-v2', 'HalfCheetah-E'),
    ('hopper-medium-v2', 'Hopper-M'),
    ('hopper-medium-replay-v2', 'Hopper-MR'),
    ('hopper-expert-v2', 'Hopper-E'),
    ('walker2d-medium-v2', 'Walker2d-M'),
    ('walker2d-medium-replay-v2', 'Walker2d-MR'),
    ('walker2d-expert-v2', 'Walker2d-E'),
]
env_groups = defaultdict(list)
for row in env_rows:
    env_groups[row['env']].append(row)
tau_sets = [{float(row['tau']) for row in env_groups[env]} for env, _ in env_labels]
if not tau_sets or any(values != tau_sets[0] for values in tau_sets[1:]):
    raise ValueError('environment sensitivity budgets are incomplete or inconsistent')
tau_env = np.array(sorted(tau_sets[0]))
env_response = {}
for env, label in env_labels:
    by_tau = {float(row['tau']): row for row in env_groups[env]}
    env_response[label] = (
        [float(by_tau[t]['dcrit_mean']) for t in tau_env],
        [float(by_tau[t]['d4rl_score_mean']) for t in tau_env],
    )

fig, axs = plt.subplots(3, 3, figsize=(7.1, 6.0), sharex=True, sharey=True,
                        constrained_layout=True)
norm = mpl.colors.Normalize(vmin=tau_env.min(), vmax=tau_env.max())
for ax, (env, (dvals, scores)) in zip(axs.flat, env_response.items()):
    dvals = np.asarray(dvals)
    scores = np.asarray(scores)
    ax.plot(dvals, scores, color='0.65', linewidth=.9, zorder=1)
    sc = ax.scatter(dvals, scores, c=tau_env, norm=norm, cmap='Greys',
                    s=22, edgecolor='black', linewidth=.25, zorder=2)
    ax.set_xscale('log')
    ax.set_xlim(.02, 1.1)
    ax.set_ylim(-6, 118)
    ax.set_title(env, fontsize=7.5)
    ax.axhline(20, color='black', linestyle=':', linewidth=.7)
    ax.grid(True, alpha=.2, linewidth=.45)
for ax in axs[-1, :]:
    ax.set_xlabel(r'$\widehat D_{\mathrm{critic}}$')
for ax in axs[:, 0]:
    ax.set_ylabel('D4RL score')
cbar = fig.colorbar(sc, ax=axs, shrink=.82, pad=.015)
cbar.set_label('total nominal budget $T$')
fig.savefig(OUT/'environment_sensitivity.pdf', bbox_inches='tight')
fig.savefig(OUT/'environment_sensitivity.png', dpi=240, bbox_inches='tight')
plt.close(fig)

# Realized critic-hop displacement and K=2/K=3 Hopper-medium frontier.
lin_hops = read_rows(
    Path('diagnostics/matched_geometry/lin2_hop_effective_step.csv'),
    ('tau', 'hop', 'own_step_sq_mean', 'action_dim'),
)
prox_runs = read_rows(
    Path('diagnostics/matched_geometry/prox2_run_diagnostics.csv'),
    ('tau', 'critic_displacement_sq_mean_metric_mean'),
)
lin_by_tau = defaultdict(list)
for row in lin_hops:
    if int(row['hop']) == 1:
        lin_by_tau[float(row['tau'])].append(float(row['own_step_sq_mean']) / int(row['action_dim']))
prox_by_tau = defaultdict(list)
for row in prox_runs:
    prox_by_tau[float(row['tau'])].append(float(row['critic_displacement_sq_mean_metric_mean']))
tau_geom = np.array(sorted(set(lin_by_tau) & set(prox_by_tau)))
if not len(tau_geom):
    raise ValueError('matched Lin-2 and Prox-2 diagnostics have no common budgets')
dcrit_exp2 = np.array([statistics.median(lin_by_tau[t]) for t in tau_geom])
dcrit_imp2 = np.array([statistics.median(prox_by_tau[t]) for t in tau_geom])

frontier_rows = read_rows(
    Path('diagnostics/matched_geometry/hopper_lin_frontier.csv'),
    ('env', 'tau', 'seed', 'total_hops', 'd4rl_score',
     'critic_displacement_sq_mean_metric_mean', 'final_displacement_sq_mean_metric_mean'),
)
frontier_groups = defaultdict(list)
for row in frontier_rows:
    if row['env'] != 'hopper-medium-v2':
        raise ValueError('hopper_lin_frontier.csv contains a non-Hopper row')
    frontier_groups[(int(row['total_hops']), float(row['tau']))].append(row)
frontier_taus = {
    hops: {total_budget for current_hops, total_budget in frontier_groups if current_hops == hops}
    for hops in (2, 3)
}
if frontier_taus[2] != frontier_taus[3]:
    raise ValueError('Hopper Lin-2 and Lin-3 frontier budgets differ')
T_hm = np.array(sorted(frontier_taus[2]))


def frontier_series(hops, field):
    values = []
    for total_budget in T_hm:
        rows = frontier_groups[(hops, float(total_budget))]
        if len(rows) != 2 or {int(row['seed']) for row in rows} != {0, 1}:
            raise ValueError(f'Hopper K={hops}, T={total_budget}: expected seeds 0 and 1')
        values.append(statistics.mean(float(row[field]) for row in rows))
    return np.array(values)


dc_k2 = frontier_series(2, 'critic_displacement_sq_mean_metric_mean')
df_k2 = frontier_series(2, 'final_displacement_sq_mean_metric_mean')
score_k2 = frontier_series(2, 'd4rl_score')
dc_k3 = frontier_series(3, 'critic_displacement_sq_mean_metric_mean')
df_k3 = frontier_series(3, 'final_displacement_sq_mean_metric_mean')
score_k3 = frontier_series(3, 'd4rl_score')

fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True)
axs[0].plot(tau_geom, dcrit_exp2, marker='D', linewidth=1.3, label='PART-Lin (K=2)')
axs[0].plot(tau_geom, dcrit_imp2, marker='s', linewidth=1.3, label='PART-Prox (K=2)')
axs[0].set_xlabel('total nominal budget $T$')
axs[0].set_ylabel(r'$\widehat D_{\mathrm{critic}}$ (median)')
axs[0].set_title('(a) Critic-coupled displacement proxy')
axs[0].grid(True, alpha=0.25, linewidth=0.5)
axs[0].legend(frameon=False, fontsize=7)

# marker size encodes score, but floor it to keep failed points visible.
size2 = 18 + 0.45*np.maximum(score_k2,0)
size3 = 18 + 0.45*np.maximum(score_k3,0)
axs[1].scatter(dc_k2, df_k2, s=size2, marker='s', label='$K=2$')
axs[1].scatter(dc_k3, df_k3, s=size3, marker='^', label='$K=3$')
for i,t in enumerate(T_hm):
    axs[1].annotate(str(int(t)), (dc_k2[i],df_k2[i]), xytext=(3,-8), textcoords='offset points', fontsize=6)
    axs[1].annotate(str(int(t)), (dc_k3[i],df_k3[i]), xytext=(3,3), textcoords='offset points', fontsize=6)
lim = 1.08
axs[1].plot([0,lim],[0,lim], linestyle='--', linewidth=0.8)
axs[1].set_xlim(0,lim); axs[1].set_ylim(0,lim)
axs[1].set_xlabel(r'$\widehat D_{\mathrm{critic}}$')
axs[1].set_ylabel(r'$\widehat D_{\mathrm{final}}$')
axs[1].set_title('(b) Hopper-medium: displacement vs. reach')
axs[1].grid(True, alpha=0.25, linewidth=0.5)
axs[1].legend(frameon=False, fontsize=7, loc='lower right')
axs[1].text(0.03, 1.01, 'labels: $T$; marker size: score', fontsize=6.5)
fig.savefig(OUT/'exposure_frontier.pdf', bbox_inches='tight')
fig.savefig(OUT/'exposure_frontier.png', dpi=240, bbox_inches='tight')
plt.close(fig)

# Stable vs collapsed diagnostic summary. Every plotted value is recomputed
# from released run-level rows; return below 20 is only the descriptive split.
lin_diagnostics = {run_key(row): row for row in read_rows(
    Path('diagnostics/matched_geometry/lin2_run_diagnostics.csv'),
    ('env', 'tau', 'seed', 'd4rl_score', 'td_residual_p99', 'final_critic_loss'),
)}
lin_movement = {run_key(row): row for row in read_rows(
    Path('diagnostics/matched_geometry/lin2_run_movement.csv'),
    ('env', 'tau', 'seed', 'critic_displacement_sq_mean_metric_mean',
     'final_displacement_sq_mean_metric_mean'),
)}
if set(lin_diagnostics) != set(lin_movement):
    raise ValueError('Lin-2 diagnostic and movement run keys differ')
lin_gain = defaultdict(float)
for row in lin_hops:
    lin_gain[run_key(row)] += float(row['canonical_q_gain_mean'])
if set(lin_gain) != set(lin_diagnostics):
    raise ValueError('Lin-2 hop and run diagnostic keys differ')

prox_diagnostics = {run_key(row): row for row in prox_runs}
prox_gain = defaultdict(float)
for row in read_rows(
    Path('diagnostics/matched_geometry/prox2_hop_geometry.csv'),
    ('env', 'tau', 'seed', 'critic_scope', 'q1_gain_hop_mean'),
):
    if row['critic_scope'] == 'canonical_td3_tau1':
        prox_gain[run_key(row)] += float(row['q1_gain_hop_mean'])
if set(prox_gain) != set(prox_diagnostics):
    raise ValueError('Prox-2 hop and run diagnostic keys differ')


def diagnostic_records(method):
    if method == 'Lin':
        records = []
        for key, row in lin_diagnostics.items():
            movement = lin_movement[key]
            records.append({
                'score': float(row['d4rl_score']),
                'td_p99': float(row['td_residual_p99']),
                'dcrit': float(movement['critic_displacement_sq_mean_metric_mean']),
                'dfinal': float(movement['final_displacement_sq_mean_metric_mean']),
                'common_gain': lin_gain[key],
            })
        return records
    records = []
    for key, row in prox_diagnostics.items():
        records.append({
            'score': float(row['d4rl_score']),
            'td_p99': float(row['td_residual_p99']),
            'dcrit': float(row['critic_displacement_sq_mean_metric_mean']),
            'dfinal': float(row['final_displacement_sq_mean_metric_mean']),
            'common_gain': prox_gain[key],
        })
    return records


labels = ['Lin stable', 'Lin collapsed', 'Prox stable', 'Prox collapsed']
groups = []
for method in ('Lin', 'Prox'):
    records = diagnostic_records(method)
    groups.append([row for row in records if row['score'] >= 20])
    groups.append([row for row in records if row['score'] < 20])
if any(not group for group in groups):
    raise ValueError('a stable/collapsed diagnostic group is empty')
td_p99 = np.array([statistics.median(row['td_p99'] for row in group) for group in groups])
dcrit = np.array([statistics.median(row['dcrit'] for row in group) for group in groups])
dfinal = np.array([statistics.median(row['dfinal'] for row in group) for group in groups])
common_gain = np.array([statistics.median(row['common_gain'] for row in group) for group in groups])

fig, axs = plt.subplots(1, 3, figsize=(7.1, 1.85), constrained_layout=True)
x=np.arange(4)
axs[0].bar(x, np.log10(td_p99))
axs[0].set_xticks(x, labels, rotation=28, ha='right', fontsize=6.5)
axs[0].set_ylabel('$\\log_{10}$ TD-error p99')
axs[0].set_title('(a) Critic failure')
axs[0].grid(True, axis='y', alpha=0.25, linewidth=0.5)

w=0.35
axs[1].bar(x-w/2,dcrit,w,label=r'$\widehat D_{\mathrm{critic}}$')
axs[1].bar(x+w/2,dfinal,w,label=r'$\widehat D_{\mathrm{final}}$')
axs[1].set_xticks(x, labels, rotation=28, ha='right', fontsize=6.5)
axs[1].set_title('(b) Realized movement')
axs[1].legend(frameon=False, fontsize=6.5)
axs[1].grid(True, axis='y', alpha=0.25, linewidth=0.5)

axs[2].bar(x,common_gain)
axs[2].axhline(0, linewidth=0.8)
axs[2].set_xticks(x, labels, rotation=28, ha='right', fontsize=6.5)
axs[2].set_ylabel('common-critic $Q$ gain')
axs[2].set_title('(c) Cross-critic validity')
axs[2].grid(True, axis='y', alpha=0.25, linewidth=0.5)
fig.savefig(OUT/'failure_signature.pdf', bbox_inches='tight')
fig.savefig(OUT/'failure_signature.png', dpi=240, bbox_inches='tight')
plt.close(fig)
