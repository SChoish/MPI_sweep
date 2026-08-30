from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 8.0
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# Common fixed-budget sweep, seed means over nine D4RL locomotion tasks.
tau = np.array([0.05,0.1,0.2,0.4,0.7,1.5,2.5,4,7,10,12,14,17,20.0])
td3 = np.array([65.19,62.81,64.85,64.22,69.14,75.43,76.80,49.16,30.35,27.14,22.08,18.51,15.35,19.21])
imp2 = np.array([61.11,66.50,68.87,70.11,74.69,78.52,81.57,72.13,64.57,44.67,30.43,30.73,24.01,27.61])
imp3 = np.array([58.31,60.69,68.61,71.32,75.25,78.21,81.55,80.91,70.84,53.91,52.60,41.47,29.23,40.51])
exp2m = np.array([60.98,63.12,66.20,72.07,67.98,72.70,80.10,73.95,45.63,39.98,27.09,25.10,24.41,20.63])
exp3m = np.array([58.32,59.97,63.69,70.52,73.23,77.14,76.94,75.31,66.47,45.78,30.65,39.54,25.74,23.88])

fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.30), constrained_layout=True)
ax = axs[0]
for y, lab, marker in [(td3,'TD3+BC','o'),(imp2,'MPI-Prox (K=2)','s'),(imp3,'MPI-Prox (K=3)','^'),(exp2m,'MPI-Lin (K=2)','D'),(exp3m,'MPI-Lin (K=3)','v')]:
    ax.plot(tau, y, marker=marker, markersize=3.2, linewidth=1.25, label=lab)
ax.set_xscale('log')
ax.set_xlabel('total nominal budget $T$')
ax.set_ylabel('mean D4RL score')
ax.set_title('(a) Stability envelope')
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.legend(frameon=False, fontsize=7, ncol=2, loc='best')

methods = ['TD3+BC','Prox-2','Prox-3','Lin-2','Lin-3']
seedmean = np.array([35,25,17,29,24])
raw = np.array([68,53,41,58,53])
seedmean_rate = seedmean / 63.0
raw_rate = raw / 126.0
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
for i,(a,b,ar,br) in enumerate(zip(seedmean,raw,seedmean_rate,raw_rate)):
    axs[1].text(i-width/2, ar+0.012, f'{a}/63', ha='center', va='bottom', fontsize=6.3)
    axs[1].text(i+width/2, br+0.012, f'{b}/126', ha='center', va='bottom', fontsize=6.3)
fig.savefig(OUT/'stability_envelope.pdf', bbox_inches='tight')
fig.savefig(OUT/'stability_envelope.png', dpi=240, bbox_inches='tight')
plt.close(fig)

# Environment-wise Prox-2 response against the current-state critic-coupled
# displacement proxy. Two-seed means from the common-batch audit at
# T in {4, 7, 10, 14, 20}.
tau_env = np.array([4, 7, 10, 14, 20.])
env_response = {
    'HalfCheetah-M': ([.033, .038, .041, .047, .056], [54.1, 57.1, 58.2, 60.5, 61.8]),
    'HalfCheetah-MR': ([.103, .111, .123, .130, .150], [49.7, 51.7, 53.1, 52.6, 54.1]),
    'HalfCheetah-ME': ([.028, .032, .040, .045, .095], [71.5, 47.8, 29.7, 5.6, 3.9]),
    'Hopper-M': ([.055, .057, .109, .435, .878], [59.5, 83.2, 50.8, 50.6, .7]),
    'Hopper-MR': ([.138, .147, .153, .167, .179], [98.9, 100.4, 99.2, 84.6, 64.7]),
    'Hopper-ME': ([.052, .106, .455, .510, .907], [101.9, 11.1, 2.7, 1.3, 1.8]),
    'Walker2d-M': ([.241, .039, .111, .759, .332], [46.3, 83.4, 16.8, 2.3, 16.7]),
    'Walker2d-MR': ([.116, .132, .149, .199, .283], [55.3, 86.4, 91.6, 19.3, 43.8]),
    'Walker2d-ME': ([.025, .048, .466, .551, .593], [112.0, 60.0, -.1, -.2, 1.0]),
}

fig, axs = plt.subplots(3, 3, figsize=(7.1, 6.0), sharex=True, sharey=True,
                        constrained_layout=True)
norm = mpl.colors.Normalize(vmin=tau_env.min(), vmax=tau_env.max())
for ax, (env, (dvals, scores)) in zip(axs.flat, env_response.items()):
    dvals = np.asarray(dvals)
    scores = np.asarray(scores)
    ax.plot(dvals, scores, color='0.65', linewidth=.9, zorder=1)
    sc = ax.scatter(dvals, scores, c=tau_env, norm=norm, cmap='viridis',
                    s=22, edgecolor='black', linewidth=.25, zorder=2)
    ax.set_xscale('log')
    ax.set_xlim(.02, 1.1)
    ax.set_ylim(-6, 118)
    ax.set_title(env, fontsize=7.5)
    ax.axhline(20, color='tab:red', linestyle=':', linewidth=.7)
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
tau_geom = np.array([4,7,10,14,20.])
dcrit_exp2 = np.array([0.0533,0.1234,0.1513,0.2350,0.3227])
dcrit_imp2 = np.array([0.0537,0.0637,0.1397,0.1666,0.1948])

T_hm = np.array([4,7,10,12,14,17,20.])
dc_k2 = np.array([0.055,0.056,0.058,0.770,0.841,0.786,0.854])
df_k2 = np.array([0.059,0.065,0.072,0.971,1.025,0.935,0.952])
score_k2 = np.array([53.45,63.36,58.12,0.66,0.66,0.69,1.28])
dc_k3 = np.array([0.054,0.055,0.056,0.352,0.125,0.257,0.746])
df_k3 = np.array([0.062,0.069,0.077,0.475,0.318,0.557,1.007])
score_k3 = np.array([64.66,74.26,85.35,50.53,41.20,7.73,0.66])

fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True)
axs[0].plot(tau_geom, dcrit_exp2, marker='D', linewidth=1.3, label='MPI-Lin (K=2)')
axs[0].plot(tau_geom, dcrit_imp2, marker='s', linewidth=1.3, label='MPI-Prox (K=2)')
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

# Stable vs collapsed diagnostic summary.
labels = ['Lin stable','Lin collapsed','Prox stable','Prox collapsed']
td_p99 = np.array([270,9.6e23,160.8,5.03e21])
qmag = np.array([311,8.48e11,346.7,6.91e10])
closs = np.array([34,1.23e23,18.66,8.82e20])
dcrit = np.array([0.112,0.538,0.064,0.446])
dfinal = np.array([0.147,0.626,0.094,0.578])
# Run-total common-critic gains for both realizations.
common_gain = np.array([2.11,-1.42,2.51,-1.06])

fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.15), constrained_layout=True)
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
