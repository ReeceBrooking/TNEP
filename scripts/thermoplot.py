import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("plots/Ethanol/Ethanol_tgap/thermo_ethanol.log")
time_ps = data[:, 1] / 1000.0   # column 2 (fs) → ps
temperature = data[:, 2]        # column 3
potential = data[:, 4]          # column 5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(time_ps, temperature)
ax1.set_xlabel("MD time (ps)")
ax1.set_ylabel("Temperature (K)")

ax2.plot(time_ps, potential)
ax2.set_xlabel("MD time (ps)")
ax2.set_ylabel("Potential energy (eV)")

plt.tight_layout()
plt.show()