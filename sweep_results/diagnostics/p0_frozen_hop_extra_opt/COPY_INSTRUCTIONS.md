First90 Hopper MART 1M checkpoints are not readable on this host.
Copy these files onto ext_csh, keeping the filenames:

  src: /home/ext_csv/mpi_sweep_lab/p0_bar_two_actor_p4_29fea94/runs/bar_p4/hopper-medium-v2_tau10_mpi4_seed0/params_1000000.pkl
  dst: sweep_results/diagnostics/p0_frozen_hop_extra_opt/source_ckpts/hopper-medium-v2_tau10_mpi4_seed0/params_1000000.pkl
  sha256: c18b5784bbb5ff656afb5d80ca61d1b5a828853cc6108b710ad009500f1d0f3d

  src: /home/ext_csv/mpi_sweep_lab/p0_bar_two_actor_p4_29fea94/runs/bar_p4/hopper-medium-v2_tau10_mpi4_seed1/params_1000000.pkl
  dst: sweep_results/diagnostics/p0_frozen_hop_extra_opt/source_ckpts/hopper-medium-v2_tau10_mpi4_seed1/params_1000000.pkl
  sha256: 6aa372aefdb40f3dac09e1f026d781d3bab7e63f4735743c358c1bda69a1acc3

  src: /home/ext_csv/mpi_sweep_lab/p0_bar_two_actor_p4_29fea94_cont2_gpu_d923/runs/bar_p4/hopper-expert-v2_tau10_mpi4_seed0/params_1000000.pkl
  dst: sweep_results/diagnostics/p0_frozen_hop_extra_opt/source_ckpts/hopper-expert-v2_tau10_mpi4_seed0/params_1000000.pkl
  sha256: 51f81441d59524dce5138520603a366434d87fa32a2e61090c2a9f51d6dcee11

  src: /home/ext_csv/mpi_sweep_lab/p0_bar_two_actor_p4_29fea94_cont2_gpu_d923/runs/bar_p4/hopper-expert-v2_tau10_mpi4_seed1/params_1000000.pkl
  dst: sweep_results/diagnostics/p0_frozen_hop_extra_opt/source_ckpts/hopper-expert-v2_tau10_mpi4_seed1/params_1000000.pkl
  sha256: 7fb0cd4a9b7259187baae09587141fe05649f34ca4769f34127c62dbe9245319

Then rerun run_p0_frozen_hop_extra_opt.py. Do not substitute other sweeps.
