from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import math
import numpy as np
import re


# -----------------------------
# Errors / parsing helpers
# -----------------------------

class InputFormatError(RuntimeError):
    pass


def input_error(msg: str, filename: str = "", line_number: int = -1) -> None:
    prefix = ""
    if filename:
        prefix += f"{filename}"
    if line_number >= 0:
        prefix += f":{line_number}"
    if prefix:
        prefix += ": "
    raise InputFormatError(prefix + msg)


def tokenize_line(line: str) -> List[str]:
    """Split by whitespace; empty -> []."""
    line = line.strip()
    if not line:
        return []
    return line.split()


def tokenize_preserve_quotes(line: str) -> List[str]:
    """
    GPUMD uses get_tokens_without_unwanted_spaces for the 2nd line.
    Extended XYZ comment lines sometimes contain lattice="...".
    We'll split on whitespace but keep quoted substrings together.
    """
    line = line.strip()
    if not line:
        return []
    # Keep quoted strings together
    return re.findall(r'''(?:[^\s"']+|"[^"]*"|'[^']*')+''', line)


def to_lower(tokens: List[str]) -> List[str]:
    return [t.lower() for t in tokens]


def parse_int(token: str, filename: str, line_number: int) -> int:
    try:
        return int(token)
    except Exception:
        input_error(f"Failed to parse int from token '{token}'.", filename, line_number)


def parse_float(token: str, filename: str, line_number: int) -> float:
    try:
        return float(token)
    except Exception:
        input_error(f"Failed to parse float from token '{token}'.", filename, line_number)


def strip_wrapping_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def parse_keyval_token(token: str, key: str) -> Optional[str]:
    """
    If token starts with f"{key}=" return the value string after '=' (raw).
    """
    k = f"{key}="
    if token.startswith(k):
        return token[len(k):]
    return None


# -----------------------------
# Geometry helpers
# -----------------------------

def get_area(a: np.ndarray, b: np.ndarray) -> float:
    # || a x b ||
    cross = np.cross(a, b)
    return float(np.linalg.norm(cross))


def get_det(box9: np.ndarray) -> float:
    # box9 is 9 elements of a 3x3 (column-major in GPUMD input, then transposed)
    m = box9.reshape(3, 3)
    return float(np.linalg.det(m))


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Parameters:
    # minimal subset required by the reader
    rc_radial_max: float
    elements: List[str]                      # allowed species symbols in order
    train_mode: int = 0                      # 0: E+F; 1: dipole; 2: pol; 3: E+F+T
    prediction: int = 0                      # 0: training; 1: prediction (relaxes some errors)
    atomic_v: int = 0                        # if atomic virial-like target used
    batch_size: int = 1
    has_bec: bool = False


@dataclass
class Structure:
    num_atom: int = 0

    # frame-level scalars
    energy: float = -1e6
    energy_weight: float = 1.0
    charge: float = 0.0
    weight: float = 1.0

    has_temperature: bool = False
    temperature: float = 0.0

    # cell and derived
    box_original: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.float32))  # 3x3 (transposed)
    box: np.ndarray = field(default_factory=lambda: np.zeros(18, dtype=np.float32))         # 0..8 cell, 9..17 inverse
    num_cell: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.int32))
    volume: float = 0.0

    # virial (6 unique comps in reduced form)
    has_virial: bool = False
    virial: np.ndarray = field(default_factory=lambda: np.full(6, -1e6, dtype=np.float32))

    # atomic properties
    type: List[int] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    z: List[float] = field(default_factory=list)

    fx: List[float] = field(default_factory=list)
    fy: List[float] = field(default_factory=list)
    fz: List[float] = field(default_factory=list)

    has_atomic_virial: bool = False
    atomic_virial_diag_only: bool = False
    avirialxx: List[float] = field(default_factory=list)
    avirialyy: List[float] = field(default_factory=list)
    avirialzz: List[float] = field(default_factory=list)
    avirialxy: List[float] = field(default_factory=list)
    avirialyz: List[float] = field(default_factory=list)
    avirialzx: List[float] = field(default_factory=list)

    has_bec: bool = False
    bec: List[float] = field(default_factory=list)  # length num_atom*9


# -----------------------------
# Box scaling (change_box)
# -----------------------------

def change_box(para: Parameters, s: Structure) -> None:
    """
    Replicates GPUMD change_box: compute replication counts num_cell
    and expanded box + inverse (stored in box[9..17]).
    """
    # In GPUMD, box_original is filled with transposed indexing.
    # We'll treat it as a 3x3 matrix in row-major for computations.
    A = np.array([s.box_original[0], s.box_original[3], s.box_original[6]], dtype=np.float64)
    B = np.array([s.box_original[1], s.box_original[4], s.box_original[7]], dtype=np.float64)
    C = np.array([s.box_original[2], s.box_original[5], s.box_original[8]], dtype=np.float64)

    det = get_det(s.box_original.astype(np.float64))
    s.volume = abs(det)

    # volume / area(b,c) gives height along a, etc.
    ha = s.volume / get_area(B, C)
    hb = s.volume / get_area(C, A)
    hc = s.volume / get_area(A, B)

    s.num_cell[0] = int(math.ceil(2.0 * para.rc_radial_max / ha))
    s.num_cell[1] = int(math.ceil(2.0 * para.rc_radial_max / hb))
    s.num_cell[2] = int(math.ceil(2.0 * para.rc_radial_max / hc))

    # expand box (first 9 entries)
    # GPUMD stores box as 18 floats: 0..8 expanded box, 9..17 inverse entries
    s.box[0] = s.box_original[0] * s.num_cell[0]
    s.box[3] = s.box_original[3] * s.num_cell[0]
    s.box[6] = s.box_original[6] * s.num_cell[0]

    s.box[1] = s.box_original[1] * s.num_cell[1]
    s.box[4] = s.box_original[4] * s.num_cell[1]
    s.box[7] = s.box_original[7] * s.num_cell[1]

    s.box[2] = s.box_original[2] * s.num_cell[2]
    s.box[5] = s.box_original[5] * s.num_cell[2]
    s.box[8] = s.box_original[8] * s.num_cell[2]

    # inverse components in the same algebraic-cofactor style as GPUMD
    s.box[9]  = s.box[4] * s.box[8] - s.box[5] * s.box[7]
    s.box[10] = s.box[2] * s.box[7] - s.box[1] * s.box[8]
    s.box[11] = s.box[1] * s.box[5] - s.box[2] * s.box[4]
    s.box[12] = s.box[5] * s.box[6] - s.box[3] * s.box[8]
    s.box[13] = s.box[0] * s.box[8] - s.box[2] * s.box[6]
    s.box[14] = s.box[2] * s.box[3] - s.box[0] * s.box[5]
    s.box[15] = s.box[3] * s.box[7] - s.box[4] * s.box[6]
    s.box[16] = s.box[1] * s.box[6] - s.box[0] * s.box[7]
    s.box[17] = s.box[0] * s.box[4] - s.box[1] * s.box[3]

    det2 = det * int(s.num_cell[0]) * int(s.num_cell[1]) * int(s.num_cell[2])
    if det2 == 0:
        input_error("Box determinant is zero; cannot invert lattice.")

    for n in range(9, 18):
        s.box[n] /= det2


# -----------------------------
# Per-atom data reading (read_force)
# -----------------------------

def read_atom_lines(
    num_columns: int,
    species_offset: int,
    pos_offset: int,
    force_offset: int,
    avirial_offset: int,
    bec_offset: int,
    lines_iter,
    para: Parameters,
    s: Structure,
    filename: str,
    line_number_ref: List[int],
    train_mode: int
) -> None:
    """
    Read N atom lines for one structure, following the offsets computed from properties=...
    line_number_ref: mutable [line_number] so we can increment in nested functions.
    """
    N = s.num_atom
    s.type = [0] * N
    s.x = [0.0] * N
    s.y = [0.0] * N
    s.z = [0.0] * N
    s.fx = [0.0] * N
    s.fy = [0.0] * N
    s.fz = [0.0] * N

    s.bec = [0.0] * (N * 9)
    if s.has_atomic_virial:
        s.avirialxx = [0.0] * N
        s.avirialyy = [0.0] * N
        s.avirialzz = [0.0] * N
        if not s.atomic_virial_diag_only:
            s.avirialxy = [0.0] * N
            s.avirialyz = [0.0] * N
            s.avirialzx = [0.0] * N

    for na in range(N):
        try:
            line = next(lines_iter)
        except StopIteration:
            input_error("Unexpected EOF while reading atom lines.", filename, line_number_ref[0])

        tokens = tokenize_line(line)
        line_number_ref[0] += 1

        if len(tokens) != num_columns:
            input_error(
                "Number of items for an atom line mismatches properties.",
                filename,
                line_number_ref[0],
            )

        atom_symbol = tokens[species_offset]

        s.x[na] = parse_float(tokens[pos_offset + 0], filename, line_number_ref[0])
        s.y[na] = parse_float(tokens[pos_offset + 1], filename, line_number_ref[0])
        s.z[na] = parse_float(tokens[pos_offset + 2], filename, line_number_ref[0])

        # forces: only if columns exist and train_mode indicates
        if num_columns > 4 and (train_mode == 0 or train_mode == 3):
            s.fx[na] = parse_float(tokens[force_offset + 0], filename, line_number_ref[0])
            s.fy[na] = parse_float(tokens[force_offset + 1], filename, line_number_ref[0])
            s.fz[na] = parse_float(tokens[force_offset + 2], filename, line_number_ref[0])

        # atomic virial-like: depends on diag-only flag
        if num_columns > 4 and s.has_atomic_virial:
            if s.atomic_virial_diag_only:
                s.avirialxx[na] = parse_float(tokens[avirial_offset + 0], filename, line_number_ref[0])
                s.avirialyy[na] = parse_float(tokens[avirial_offset + 1], filename, line_number_ref[0])
                s.avirialzz[na] = parse_float(tokens[avirial_offset + 2], filename, line_number_ref[0])
            else:
                # 9 entries present, but we keep 6 reduced components
                s.avirialxx[na] = parse_float(tokens[avirial_offset + 0], filename, line_number_ref[0])
                s.avirialyy[na] = parse_float(tokens[avirial_offset + 4], filename, line_number_ref[0])
                s.avirialzz[na] = parse_float(tokens[avirial_offset + 8], filename, line_number_ref[0])
                s.avirialxy[na] = parse_float(tokens[avirial_offset + 3], filename, line_number_ref[0])
                s.avirialyz[na] = parse_float(tokens[avirial_offset + 7], filename, line_number_ref[0])
                s.avirialzx[na] = parse_float(tokens[avirial_offset + 6], filename, line_number_ref[0])

        # bec: 9 floats
        if num_columns > 4 and s.has_bec:
            for d in range(9):
                s.bec[na * 9 + d] = parse_float(tokens[bec_offset + d], filename, line_number_ref[0])

        # map atom symbol to type index
        try:
            t = para.elements.index(atom_symbol)
        except ValueError:
            input_error(
                "There is atom in train.xyz or test.xyz that are not in nep.in.",
                filename,
                line_number_ref[0],
            )
        s.type[na] = t


# -----------------------------
# Frame header parsing (read_one_structure)
# -----------------------------

def parse_lattice(tokens: List[str], s: Structure, filename: str, line_number: int, para: Parameters) -> None:
    has_lattice = False
    lattice_key = "lattice="

    for n, tok in enumerate(tokens):
        if tok.startswith(lattice_key):
            has_lattice = True

            # GPUMD expects 9 numbers after lattice= as separate tokens.
            # First token is lattice="a b c ..."? Here they treat them as tokens[n+m]
            # with quotes around first and last.
            transpose_index = [0, 3, 6, 1, 4, 7, 2, 5, 8]
            for m in range(9):
                t = tokens[n + m]
                if m == 0:
                    # lattice="x
                    raw = t[len(lattice_key):]
                    raw = raw[1:] if raw.startswith('"') else raw
                elif m == 8:
                    raw = t[:-1] if t.endswith('"') else t
                else:
                    raw = t
                s.box_original[transpose_index[m]] = parse_float(raw, filename, line_number)

            change_box(para, s)
            break

    if not has_lattice:
        input_error("'lattice' is missing in the second line of a frame.", filename, line_number)


def parse_energy(tokens: List[str], s: Structure, filename: str, line_number: int, para: Parameters) -> None:
    has_energy = False
    for tok in tokens:
        v = parse_keyval_token(tok, "energy")
        if v is not None:
            has_energy = True
            s.energy = parse_float(v, filename, line_number) / s.num_atom
    if (para.train_mode == 0 or para.train_mode == 3) and not has_energy:
        input_error("'energy' is missing in the second line of a frame.", filename, line_number)


def parse_energy_weight(tokens: List[str], s: Structure, filename: str, line_number: int) -> None:
    for tok in tokens:
        v = parse_keyval_token(tok, "energy_weight")
        if v is not None:
            s.energy_weight = parse_float(v, filename, line_number)


def parse_charge(tokens: List[str], s: Structure, filename: str, line_number: int) -> None:
    for tok in tokens:
        v = parse_keyval_token(tok, "charge")
        if v is not None:
            s.charge = parse_float(v, filename, line_number)


def parse_temperature(tokens: List[str], s: Structure, filename: str, line_number: int, para: Parameters) -> None:
    s.has_temperature = False
    for tok in tokens:
        v = parse_keyval_token(tok, "temperature")
        if v is not None:
            s.has_temperature = True
            s.temperature = parse_float(v, filename, line_number)

    if para.train_mode == 3 and not s.has_temperature:
        input_error("'temperature' is missing in the second line of a frame.", filename, line_number)

    if not s.has_temperature:
        s.temperature = 0.0


def parse_weight(tokens: List[str], s: Structure, filename: str, line_number: int) -> None:
    s.weight = 1.0
    for tok in tokens:
        v = parse_keyval_token(tok, "weight")
        if v is not None:
            s.weight = parse_float(v, filename, line_number)
            if s.weight <= 0.0 or s.weight > 100.0:
                input_error("Configuration weight should > 0 and <= 100.", filename, line_number)


def parse_virial_or_stress(tokens: List[str], s: Structure, filename: str, line_number: int, para: Parameters) -> None:
    # virial= : reduced_index mapping
    reduced_index = [0, 3, 5, 3, 1, 4, 5, 4, 2]

    has_virial = False
    for n, tok in enumerate(tokens):
        if tok.startswith("virial="):
            has_virial = True
            s.has_virial = True
            for m in range(9):
                t = tokens[n + m]
                if m == 0:
                    raw = t[len("virial="):]
                    raw = raw[1:] if raw.startswith('"') else raw
                elif m == 8:
                    raw = t[:-1] if t.endswith('"') else t
                else:
                    raw = t
                s.virial[reduced_index[m]] = parse_float(raw, filename, line_number) / s.num_atom

    # stress= : convert to virial per atom using volume
    has_stress = False
    virials_from_stress = np.zeros(6, dtype=np.float64)
    for n, tok in enumerate(tokens):
        if tok.startswith("stress="):
            has_stress = True
            volume = abs(get_det(s.box_original.astype(np.float64)))
            for m in range(9):
                t = tokens[n + m]
                if m == 0:
                    raw = t[len("stress="):]
                    raw = raw[1:] if raw.startswith('"') else raw
                elif m == 8:
                    raw = t[:-1] if t.endswith('"') else t
                else:
                    raw = t
                virials_from_stress[reduced_index[m]] = parse_float(raw, filename, line_number) * (-volume / s.num_atom)

    if s.has_virial and has_stress:
        tol = 1e-3
        for m in range(6):
            if abs(float(s.virial[m]) - float(virials_from_stress[m])) > tol:
                if para.prediction == 0:
                    input_error("Virials and stresses for structure are inconsistent!", filename, line_number)
        # if training, keep virial; if prediction, still keep virial
    elif (not s.has_virial) and has_stress:
        for m in range(6):
            s.virial[m] = float(virials_from_stress[m])
        s.has_virial = True

    if not s.has_virial:
        s.virial[:] = -1e6


def parse_dipole_or_pol(tokens: List[str], s: Structure, filename: str, line_number: int, para: Parameters) -> None:
    reduced_index = [0, 3, 5, 3, 1, 4, 5, 4, 2]

    # train_mode == 1: dipole= (stored in virial[0:3], per atom)
    if para.train_mode == 1:
        s.has_virial = False
        for n, tok in enumerate(tokens):
            if tok.startswith("dipole="):
                s.has_virial = True
                s.virial[:] = 0.0
                for m in range(3):
                    t = tokens[n + m]
                    if m == 0:
                        raw = t[len("dipole="):]
                        raw = raw[1:] if raw.startswith('"') else raw
                    elif m == 2:
                        raw = t[:-1] if t.endswith('"') else t
                    else:
                        raw = t
                    s.virial[m] = parse_float(raw, filename, line_number) / s.num_atom
        if not s.has_virial:
            if para.prediction == 0:
                input_error("'dipole' is missing in the second line of a frame.", filename, line_number)
            else:
                s.virial[:] = -1e6

    # train_mode == 2: pol= (stored in virial[0..5], reduced, per atom)
    if para.train_mode == 2:
        s.has_virial = False
        for n, tok in enumerate(tokens):
            if tok.startswith("pol="):
                s.has_virial = True
                for m in range(9):
                    t = tokens[n + m]
                    if m == 0:
                        raw = t[len("pol="):]
                        raw = raw[1:] if raw.startswith('"') else raw
                    elif m == 8:
                        raw = t[:-1] if t.endswith('"') else t
                    else:
                        raw = t
                    s.virial[reduced_index[m]] = parse_float(raw, filename, line_number) / s.num_atom
        if not s.has_virial:
            if para.prediction == 0:
                input_error("'pol' is missing in the second line of a frame.", filename, line_number)
            else:
                s.virial[:] = -1e6


def parse_properties(tokens: List[str], s: Structure, para: Parameters, filename: str, line_number: int) -> Tuple[int, int, int, int, int, int]:
    """
    Parse properties=... specification to determine offsets and flags.
    Returns: num_columns, species_offset, pos_offset, force_offset, avirial_offset, bec_offset
    """
    species_offset = pos_offset = force_offset = avirial_offset = bec_offset = 0
    num_columns = 0
    s.has_atomic_virial = False
    s.atomic_virial_diag_only = False
    s.has_bec = False

    for n, tok in enumerate(tokens):
        if tok.startswith("properties="):
            # GPUMD replaces ':' with space, then tokenizes
            line = tok[len("properties="):]
            line = line.replace(":", " ")
            sub_tokens = tokenize_line(line)

            # GPUMD expects triplets: name, type, count
            if len(sub_tokens) % 3 != 0:
                input_error("Invalid properties= format.", filename, line_number)

            # positions of fields
            species_pos = pos_pos = force_pos = avirial_pos = bec_pos = -1

            for k in range(len(sub_tokens) // 3):
                name = sub_tokens[k * 3].lower()
                if name == "species":
                    species_pos = k
                if name == "pos":
                    pos_pos = k
                if name in ("force", "forces"):
                    force_pos = k
                if name in ("adipole", "atomic_dipole"):
                    avirial_pos = k
                    s.has_atomic_virial = True
                    s.atomic_virial_diag_only = True
                if name in ("apol", "atomic_polarizability"):
                    avirial_pos = k
                    s.has_atomic_virial = True
                    s.atomic_virial_diag_only = False
                if name == "bec":
                    bec_pos = k
                    s.has_bec = True
                    para.has_bec = True

            if species_pos < 0:
                input_error("'species' is missing in properties.", filename, line_number)
            if pos_pos < 0:
                input_error("'pos' is missing in properties.", filename, line_number)
            if force_pos < 0 and (para.train_mode == 0 or para.train_mode == 3):
                input_error("'force' or 'forces' is missing in properties.", filename, line_number)
            if avirial_pos < 0 and para.train_mode == 1 and para.atomic_v == 1:
                input_error("'adipole' or 'atomic_dipole' is missing in properties.", filename, line_number)
            if avirial_pos < 0 and para.train_mode == 2 and para.atomic_v == 1:
                input_error("'apol' or 'atomic_polarizability' is missing in properties.", filename, line_number)

            # offsets: sum of counts of prior fields
            def count_at(k: int) -> int:
                return parse_int(sub_tokens[k * 3 + 2], filename, line_number)

            for k in range(len(sub_tokens) // 3):
                c = count_at(k)
                if k < species_pos:
                    species_offset += c
                if k < pos_pos:
                    pos_offset += c
                if force_pos >= 0 and k < force_pos:
                    force_offset += c
                if avirial_pos >= 0 and k < avirial_pos:
                    avirial_offset += c
                if bec_pos >= 0 and k < bec_pos:
                    bec_offset += c
                num_columns += c

            break  # only one properties= token handled

    return num_columns, species_offset, pos_offset, force_offset, avirial_offset, bec_offset


def read_one_structure(
    para: Parameters,
    lines_iter,
    s: Structure,
    filename: str,
    line_number_ref: List[int],
) -> None:
    """
    Read the 2nd line of a frame + the subsequent N atom lines.
    """
    try:
        comment_line = next(lines_iter)
    except StopIteration:
        input_error("Unexpected EOF reading structure header.", filename, line_number_ref[0])

    tokens = tokenize_preserve_quotes(comment_line)
    line_number_ref[0] += 1
    tokens = to_lower(tokens)

    if len(tokens) == 0:
        input_error("The second line for each frame should not be empty.", filename, line_number_ref[0])

    # Scalars
    parse_energy_weight(tokens, s, filename, line_number_ref[0])
    parse_energy(tokens, s, filename, line_number_ref[0], para)
    parse_charge(tokens, s, filename, line_number_ref[0])
    parse_temperature(tokens, s, filename, line_number_ref[0], para)
    parse_weight(tokens, s, filename, line_number_ref[0])

    # Lattice (required)
    parse_lattice(tokens, s, filename, line_number_ref[0], para)

    # virial/stress (optional)
    parse_virial_or_stress(tokens, s, filename, line_number_ref[0], para)

    # dipole / pol stored in virial slot depending on train_mode
    parse_dipole_or_pol(tokens, s, filename, line_number_ref[0], para)

    # properties parsing -> offsets + flags
    (num_columns, species_offset, pos_offset, force_offset, avirial_offset, bec_offset) = parse_properties(
        tokens, s, para, filename, line_number_ref[0]
    )

    # Now read N atom lines
    read_atom_lines(
        num_columns=num_columns,
        species_offset=species_offset,
        pos_offset=pos_offset,
        force_offset=force_offset,
        avirial_offset=avirial_offset,
        bec_offset=bec_offset,
        lines_iter=lines_iter,
        para=para,
        s=s,
        filename=filename,
        line_number_ref=line_number_ref,
        train_mode=para.train_mode,
    )


# -----------------------------
# Reading the whole exyz file
# -----------------------------

def read_exyz(
    para: Parameters,
    path: str
) -> List[Structure]:
    structures: List[Structure] = []
    line_number_ref = [0]

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines_iter = iter(f)

        Nc = 0
        while True:
            # First line of frame: num atoms
            try:
                line = next(lines_iter)
            except StopIteration:
                break

            line_number_ref[0] += 1
            tokens = tokenize_line(line)

            if len(tokens) == 0:
                break
            if len(tokens) > 1:
                input_error("The first line for each frame should have one value.", path, line_number_ref[0])

            s = Structure()
            s.num_atom = parse_int(tokens[0], path, line_number_ref[0])
            if s.num_atom < 1:
                input_error("Number of atoms for each frame should >= 1.", path, line_number_ref[0])

            read_one_structure(para, lines_iter, s, path, line_number_ref)
            structures.append(s)
            Nc += 1

    print(f"Number of configurations = {Nc}.")

    # warning about very negative energies (as in GPUMD)
    for s in structures:
        if s.energy < -100.0:
            print("Warning:")
            print("    There is energy < -100 eV/atom in the data set.")
            print("    Because we use single precision in NEP training")
            print("    it means that the reference and calculated energies")
            print("    might only be accurate up to 1 meV/atom")
            print("    which can effectively introduce noises.")
            print("    We suggest you preprocess (using double precision)")
            print("    your data to make the energies closer to 0.")
            break

    return structures


# -----------------------------
# Reordering by energy into batches
# -----------------------------

def find_permuted_indices(num_batches: int, structures: List[Structure]) -> List[int]:
    energies = [s.energy for s in structures]
    energy_index = list(range(len(structures)))
    energy_index.sort(key=lambda i: energies[i])  # stable sort in python

    permuted = [0] * len(structures)
    count = 0
    for b in range(num_batches):
        batch_min = len(structures) // num_batches
        is_larger = b + batch_min * num_batches < len(structures)
        batch_size = batch_min + 1 if is_larger else batch_min

        for c in range(batch_size):
            permuted[count + c] = energy_index[b + num_batches * c]
        count += batch_size

    return permuted


def reorder(num_batches: int, structures: List[Structure]) -> None:
    configuration_id = find_permuted_indices(num_batches, structures)
    copy = [structures[i] for i in range(len(structures))]  # shallow ok; we replace entries

    # Rebuild list in-place with deep-ish copy of per-atom arrays
    new_structs: List[Structure] = []
    for nc in range(len(structures)):
        src = copy[configuration_id[nc]]
        dst = Structure()
        # frame-level
        dst.num_atom = src.num_atom
        dst.weight = src.weight
        dst.has_virial = src.has_virial
        dst.energy = src.energy
        dst.energy_weight = src.energy_weight
        dst.has_temperature = src.has_temperature
        dst.temperature = src.temperature
        dst.volume = src.volume

        dst.virial = np.array(src.virial, dtype=np.float32).copy()
        dst.box = np.array(src.box, dtype=np.float32).copy()
        dst.box_original = np.array(src.box_original, dtype=np.float32).copy()
        dst.num_cell = np.array(src.num_cell, dtype=np.int32).copy()

        # per-atom
        dst.type = list(src.type)
        dst.x = list(src.x)
        dst.y = list(src.y)
        dst.z = list(src.z)
        dst.fx = list(src.fx)
        dst.fy = list(src.fy)
        dst.fz = list(src.fz)

        dst.has_atomic_virial = src.has_atomic_virial
        dst.atomic_virial_diag_only = src.atomic_virial_diag_only
        dst.avirialxx = list(src.avirialxx)
        dst.avirialyy = list(src.avirialyy)
        dst.avirialzz = list(src.avirialzz)
        dst.avirialxy = list(src.avirialxy)
        dst.avirialyz = list(src.avirialyz)
        dst.avirialzx = list(src.avirialzx)

        dst.has_bec = src.has_bec
        dst.bec = list(src.bec)

        new_structs.append(dst)

    structures[:] = new_structs


# -----------------------------
# Top-level convenience function
# -----------------------------

def read_structures(is_train: bool, para: Parameters) -> Tuple[List[Structure], bool]:
    path = "train.xyz" if is_train else "test.xyz"
    has_test_set = True

    try:
        structures = read_exyz(para, path)
    except FileNotFoundError:
        if is_train:
            input_error("Failed to open train.xyz.", path, 0)
        else:
            has_test_set = False
            structures = []

    # reorder if training and batch_size < num_structures and prediction==0
    if para.prediction == 0 and is_train and para.batch_size < len(structures):
        num_batches = (len(structures) - 1) // para.batch_size + 1
        reorder(num_batches, structures)

    return structures, has_test_set
