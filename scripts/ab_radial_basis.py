"""A/B: fixed SOAP basis vs learned radial basis on the CHO dipole set.
Usage: python scripts/ab_radial_basis.py <K or none> <num_generations> [total_N]
Prints the run's own terminal output; nothing is post-processed."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MasterTNEP import train_model
from TNEPconfig import TNEPconfig

K = None if sys.argv[1].lower() == "none" else int(sys.argv[1])
cfg = TNEPconfig()
cfg.data_path = "datasets/train.xyz"
cfg.test_data_path = "datasets/test.xyz"
cfg.allowed_species = [6, 1, 8]
cfg.descriptor_mode = 1
cfg.radial_basis_size = K
cfg.num_generations = int(sys.argv[2])
cfg.total_N = int(sys.argv[3]) if len(sys.argv) > 3 else None
cfg.seed = 928375439201
cfg.save_path = f"models/ab_radial_K{K}"
cfg.checkpoint_interval = None
cfg.plot_interval = None
cfg.show_plots = False
train_model(cfg)
