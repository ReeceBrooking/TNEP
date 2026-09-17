"""Learned SOAP-turbo radial basis (cfg.radial_basis_size)."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest
import tensorflow as tf
from ase.io import read

from TNEPconfig import TNEPconfig
from DescriptorBuilderGPU import compute_dim_q

CHO_XYZ = os.path.join(ROOT, "datasets", "train.xyz")


def _cho_frames(n=3):
    """First n CHO-only frames of train.xyz (C=6, O=8, H=1)."""
    out = []
    for at in read(CHO_XYZ, index=":40"):
        if set(at.numbers).issubset({6, 8, 1}):
            out.append(at)
        if len(out) == n:
            break
    return out


def _cfg(**kw):
    cfg = TNEPconfig()
    cfg.target_mode = 1
    cfg.dipole_readout = "neighbor"
    cfg.descriptor_mode = 1
    cfg.l_max = 4
    cfg.alpha_max = 4
    cfg.rcut_hard = 4.0
    cfg.rcut_soft = 3.5
    cfg.dipole_rc = 3.5
    cfg.radial_enhancement = 0
    cfg.num_neurons = 8
    cfg.activation = "tanh"
    cfg.descriptor_mixing = False
    cfg.descriptor_preprocess_contract = "off"
    cfg.dipole_channels = 4
    cfg.dipole_species_pairs = "off"
    cfg.dipole_radial = "chebyshev"
    cfg.dipole_cutoff = None
    cfg.seed = 7
    cfg.types = [6, 8, 1]
    cfg.num_types = 3
    cfg.type_map = {6: 0, 8: 1, 1: 2}
    cfg.radial_basis_size = None
    for k, v in kw.items():
        setattr(cfg, k, v)
    cfg.dim_q = compute_dim_q(cfg)
    return cfg


def test_gpu_builder_self_only_neighbor_readout_emits_no_gradients():
    from DescriptorBuilder import make_descriptor_builder
    cfg = _cfg()
    b = make_descriptor_builder(cfg)
    descs, grads, gidx = b.build_descriptors_self_only(_cho_frames(2), batch_size=None)
    assert len(descs) == 2 and descs[0].shape[1] == cfg.dim_q
    assert all(g.shape[0] == 0 for g in grads[0])   # neighbor readout reads no gradients
    assert gidx[0] == [[] for _ in range(len(descs[0]))]


def test_zero_pair_data_stages_on_both_paths():
    """P = 0 (neighbor readout, GPU builder) is a supported invariant through
    pad_and_stack and both chunk-staging paths."""
    from data import (prepare_eval_data, prefetched_chunks, move_data_to_gpu,
                      prestage_chunk_indices, slice_and_complete_chunk)
    cfg = _cfg()
    data = prepare_eval_data(_cho_frames(3), cfg)
    assert tuple(data["grad_values"].shape) == (0, 3, cfg.dim_q)
    ranges = [(0, 2), (2, 3)]
    prestage_chunk_indices(data, ranges)
    for pin in (True, False):
        d = dict(data)
        if not pin:
            d["_gv_resident_gpu"] = True
            move_data_to_gpu(d)
        chunks = [c for _, _, c in prefetched_chunks(d, ranges, pin_to_cpu=pin)]
        assert [int(c["descriptors"].shape[0]) for c in chunks] == [2, 1]
        assert all(int(c["grad_values"].shape[0]) == 0 for c in chunks)
    c = slice_and_complete_chunk(data, [1, 2])
    assert int(c["pair_struct"].shape[0]) == 0
