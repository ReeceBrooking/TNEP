"""Short end-to-end run across target modes and preconditioning options.

    MODE=2 OPT=weight_reparam STAT=std python scripts/smoke_modes.py

Env: MODE (0/1/2), OPT (sigma_scaling|weight_reparam), STAT (std|rms|cv),
     PRE (descriptor_preprocess_contract, to exercise that guard),
     N (structures; unset = whole dataset), GENS, OUT.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TNEPconfig import TNEPconfig                       # noqa: E402
from MasterTNEP import train_model                      # noqa: E402

cfg = TNEPconfig()
cfg.target_mode = int(os.environ.get("MODE", 2))
if cfg.target_mode == 0:
    cfg.data_path, cfg.test_data_path = "datasets/PEStrain.xyz", None
    cfg.allowed_species = None
opt, stat = os.environ.get("OPT"), os.environ.get("STAT", "std")
if opt:
    setattr(cfg, f"descriptor_{opt}", stat)
if os.environ.get("PRE"):                       # exercise the preprocess guard
    cfg.descriptor_preprocess_contract = os.environ["PRE"]
# N unset (or empty) means the WHOLE dataset. total_N=0 is not "no limit":
# TNEPconfig.randomise truncates to zero structures and build_and_reduce then
# raises on the empty dataset.
_n = os.environ.get("N")
cfg.total_N = int(_n) if _n else None
cfg.num_generations = int(os.environ.get("GENS", 6))
cfg.val_interval = 2
cfg.checkpoint_interval = 10 ** 9
cfg.save_plots = None                           # str | None, not bool
cfg.show_plots = False
cfg.save_path = os.environ.get("OUT", "models/smoke")
train_model(cfg=cfg)
print(f"OK mode={cfg.target_mode} opt={opt or 'off'} stat={stat}")
