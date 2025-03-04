import numpy as np
import pandas as pd
import random
from multiprocessing import Pool

# This file provides the code necessary to calculate the memory capacity when a single node is lesioned.
# The file can be adapted for plasticity and delays by following the tutorial in "Plasticity_and_delay_tutorial.ipynb"

import os
import sys

from conn2res.connectivity import Conn
import RC_lesion as RC
import task_class as tc
from conn2res import readout

PROJ_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJ_DIR, 'data', 'human')

# Define the necessary parameters (based on parameter search)
INPUT_GAIN = 1
ALPHA = 0.9
TARGET = 0.0

# Load the necessary files
COORDS = np.load(os.path.join(DATA_DIR, 'coords.npy')) # Only for delays
RSN_MAPPING = np.load(os.path.join(DATA_DIR, 'rsn_mapping.npy')) # To define input and output nodes

# Define the connectome
file = f'data/human/consensus_0.npy' # Publicly available data of Suárez et al.: https://doi.org/10.5281/zenodo.10205004

w = np.load(file)
conn = Conn(w=w) # Buil the reservoir using Conn2res
conn.scale_and_normalize() # Scale the reservoir weights between 0 and 1 ; normalise by the spectral radius

# Determine input and output nodes
input_nodes = conn.get_nodes(nodes_from=None, node_set='subctx')
output_nodes = conn.get_nodes(nodes_from = None, node_set = 'VIS')

# Build the input matrix
w_in = np.zeros((1, conn.n_nodes)) # The size of the input matrix depends on the number of input channels
w_in[:, input_nodes] = np.eye(1)

# Load task
data = np.load(f"tasks/memory_task_delay_0.npz") # This task was created in "create_task.py"
u_train, u_test, y_train, y_test = data["u_train"], data["u_test"], data["y_train"], data["y_test"]

# Lesion each node once, excpet for the input and output nodes
nodes_lesion = list(set(np.arange(0, 1015)) - set(input_nodes) - set(output_nodes))

def run_simulation(node):
    # Adapt this for plasticity and delays
    esn = RC.ESNLesion(w=conn.w, targets = np.repeat(TARGET, conn.w.shape[0]), activation_function='tanh', n_lesions=[node], input_nodes=input_nodes, output_nodes=output_nodes, plasticity=False)
    readout_module = tc.ReadoutMemory(estimator=readout.select_model(y_train), tau=20)
    esn.w = ALPHA * conn.w
    
    rs_train = esn.simulate(ext_input=u_train, w_in=w_in, output_nodes=output_nodes)
    rs_test = esn.simulate(ext_input=u_test, w_in=w_in, output_nodes=output_nodes)
    MC_train, MC_test, e_train, e_test = readout_module.run_task(X=(rs_train, rs_test), y=(y_train, y_test))
    return {"n_lesions": node, "MC": MC_test}

if __name__ == "__main__":
    # Optional: parallel processing
    with Pool(processes=60) as pool:
        results = pool.map(run_simulation, nodes_lesion)

    # Save results in a .csv file
    pd.DataFrame(results).to_csv(f"SingleNode_subctx_VIS.csv", index=False)
