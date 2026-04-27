import qml, pickle

data, = qml.data.load("qchem", molname="H2O", bondlength="full", 
                       basis="STO-3G", 
                       attributes=["fci_energy", "dipole_op", "molecule", "number_op"])

pickle.dump(data, open("h2o_data.pkl", "wb"))
print("Saved.")