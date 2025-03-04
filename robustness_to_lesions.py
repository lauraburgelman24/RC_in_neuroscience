# This file describes the experiment that investigates the robustness to lesions
# Plasticity and delays can be added following the guidelines in "Plasticity_and_delay_tutorial.ipynb"

import numpy as np
import pandas as pd
import random
from multiprocessing import Pool

import os
import sys

from conn2res.connectivity import Conn
import RC_lesion as RC
import task_class as tc
from conn2res import readout

PROJ_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJ_DIR, 'data', 'human')

# Define the necessary parameters
INPUT_GAIN = 1
ALPHA = 0.9 # Alpha should be determined in a parameter search

# Load the necessary files
COORDS = np.load(os.path.join(DATA_DIR, 'coords.npy')) # Only necessary for delays
RSN_MAPPING = np.load(os.path.join(DATA_DIR, 'rsn_mapping.npy')) # Necessary to define input and output nodes

# Define the connectome
file = f'data/human/consensus_0.npy' # Publicly available connectome: https://doi.org/10.5281/zenodo.10205004

w = np.load(file)
conn = Conn(w=w) # Use Conn2res to build a reservoir
conn.scale_and_normalize() # Scale the reservoir weights between 0 and 1 ; normalise by the spectral radius

# Determine input and output nodes
input_nodes = conn.get_nodes(nodes_from = None, node_set = "subctx")
output_nodes = conn.get_nodes(nodes_from = None, node_set = 'SM')

# Build the input matrix
w_in = np.zeros((1, conn.n_nodes)) # The size of the matrix depends on the number of input channels
w_in[:, input_nodes] = np.eye(1) # Only the input nodes are "stimulated" by the input

# Load task
data = np.load(f"tasks/memory_task_0.npz") # This task was created in a separate file "create_task.py"
u_train, u_test, y_train, y_test = data["u_train"], data["u_test"], data["y_train"], data["y_test"]

# Build a dataframe that holds the MC for different lesions
lesions = np.linspace(0, 780, 40) # We gradually increase the number of lesions
l = []
identifier = 0
for lesion in lesions:
    for i in range(100): # We create a sample with N = 100
        d = {"n_lesions": lesion, "identifier": identifier}
        l.append(d)
        identifier += 1
parameters = pd.DataFrame.from_dict(l)

def run_simulation(parameters):
    lesion, identifier = parameters
    
    # This holds for a model without plasticity or delays and should otherwise be updated
    
    esn = RC.ESNLesion(w=conn.w, targets = np.repeat(0.0, conn.w.shape[0]), activation_function='tanh', n_lesions=int(lesion), input_nodes=input_nodes, output_nodes=output_nodes, plasticity=False) # Use the attribute n_lesions to specify the number of lesioned nodes (random choice)
    readout_module = tc.ReadoutMemory(estimator=readout.select_model(y_train), tau=20)
    esn.w = ALPHA * conn.w
    
    rs_train = esn.simulate(ext_input=u_train, w_in=w_in, output_nodes=output_nodes)
    rs_test = esn.simulate(ext_input=u_test, w_in=w_in, output_nodes=output_nodes)
    MC_train, MC_test, e_train, e_test = readout_module.run_task(X=(rs_train, rs_test), y=(y_train, y_test))
    return {"n_lesions": lesion, "MC": MC_test}

if __name__ == "__main__":
    # Optional: parallel processing
    with Pool(processes=25) as pool:
        results = pool.map(run_simulation, parameters.values)

    # Write the results to a .csv file
    pd.DataFrame(results).to_csv(f"dataframes/df_subctx_np_VIS.csv", index=False)
