import numpy as np
import pandas as pd

import os
import sys
import glob
import re

from multiprocessing import Pool
from conn2res.connectivity import Conn
import RC_lesion as RC
import task_class as tc
from conn2res import readout

PROJ_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJ_DIR, 'data', 'human')

# Set the necessary parameters
INPUT_GAIN = 1
ALPHA = 0.95 # Identified through parameter search

# determine input and output nodes
input_nodes = [155, 163] # Thalamic nodes
output_nodes = [4, 10, 18, 19, 20, 21, 41, 43, 50, 57, 58, 64, 78, 84, 93, 94, 95, 115, 117, 124, 131, 132, 134] # Visual nodes identified using overlapping atlases in FSL

# Load the task
data = np.load(f"tasks/memory_task_0.npz") # The task was created using a separate file "create_task.py"
u_train, u_test, y_train, y_test = data["u_train"], data["u_test"], data["y_train"], data["y_test"]

# Gather files patient and control connectomes
files = sorted(glob.glob("data/human/C_*"))
ids = [int(re.search(r'C_(\d+)', f).group(1)) for f in files] # Store identifier
l = []
for i, file in zip(ids, files):
    d = {'identifier': i, 'file_name': file}
    l.append(d)
parameters = pd.DataFrame.from_dict(l)

def run_simulation(parameter):
    identifier, file_name = parameter

    # Load connectome
    w = np.load(os.path.join(file_name))
    conn = Conn(w=w) # Build reservoir from connectome
    conn.scale_and_normalize() # Scale reservoir weights between 0 and 1 ; normalise by the spectral radius

    # Build the input matrix
    w_in = np.zeros((1, conn.n_nodes)) # The size of the input matrix depends on the number of input channels
    w_in[:, input_nodes] = np.eye(1)

    # This part can be adapted to include plasticity and/or delays
    esn = RC.ESNLesion(w=conn.w, targets = np.repeat(0.0, conn.w.shape[0]), activation_function='tanh', n_lesions=None, input_nodes=input_nodes, output_nodes=output_nodes, plasticity=False)
    
    readout_module = tc.ReadoutMemory(estimator=readout.select_model(y_train), tau=20)
    esn.w = ALPHA * conn.w
    
    rs_train = esn.simulate(ext_input=u_train, w_in=w_in, output_nodes=output_nodes)
    rs_test = esn.simulate(ext_input=u_test, w_in=w_in, output_nodes=output_nodes)
    MC_train, MC_test, e_train, e_test = readout_module.run_task(X=(rs_train, rs_test), y=(y_train, y_test))
    return {"identifier": identifier, "MC": MC_test}

if __name__ == "__main__":
    # Optional: parallel processing
    with Pool(processes=45) as pool:
        results = pool.map(run_simulation, parameters.values)
    # Write to a .csv file
    pd.DataFrame(results).to_csv(f"df_clinical.csv", index=False)
