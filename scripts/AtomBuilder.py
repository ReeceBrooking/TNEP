from ase.build import molecule
from ase.io import write

molecule = molecule('CH3CH2OH')   # or 'C2H5OH'
molecule.center(vacuum=20.0)   # 20 Å of vacuum on all sides
molecule.pbc = True
write('ethanol.xyz', molecule)