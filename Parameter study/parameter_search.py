import numpy as np
import pandas as pd
import random
import os
import sys
from multiprocessing import Pool
import time
import psutil

from conn2res.connectivity import Conn
import RC_lesion as RC
import task_class as tc
from conn2res import readout

# N determines which set of parameters to use
N = sys.argv[sys.argv.index('--N') + 1]

PROJ_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJ_DIR, 'data', 'human')

# Load the task
data = np.load(f"tasks/memory_task_delay_{N}.npz") # The task(s) were created using "create_task.py"

u_train = data['u_train']
u_test = data['u_test']
y_train = data['y_train']
y_test = data['y_test']

# Load the parameter combinations
params = pd.read_csv(f"params/param_samples_{N}.csv")

# Load the necessary files
COORDS = np.load(os.path.join(DATA_DIR, 'coords.npy')) # Only for delays

# Define the connectome
file = f'data/human/consensus_0.npy'

w = np.load(file)
conn = Conn(w=w) # Build a reservoir from the connectome
conn.scale_and_normalize() # Scale the reservoir weights between 0 and 1 ; normalise by the spectral radius

# Determine input and output nodes
input_nodes = conn.get_nodes(nodes_from=None, node_set='DA')
output_nodes = conn.get_nodes(nodes_from=None, node_set='SM')

# Build the input matrix
w_in = np.zeros((1, conn.n_nodes)) # The size of the input matrix is determined by the number of input channels
w_in[:, input_nodes] = np.eye(1)

def run_simulation(alpha_dtv):
    # Load the parameters
    alpha, target = alpha_dtv
    
    readout_module = tc.ReadoutMemory(estimator=readout.select_model(y_train), tau=20)

    # Adapt this for plasticity and/or delays
    esn = RC.ESNLesion(w=conn.w, targets=np.repeat(target, conn.w.shape[0]), activation_function='tanh', input_nodes=input_nodes, output_nodes=output_nodes, plasticity=False)

    esn.w = alpha * conn.w

    rs_train = esn.simulate(
        ext_input=u_train, w_in=w_in,
        output_nodes=output_nodes
    )  # generating the reservoir dynamics based on the input

    rs_test = esn.simulate(
        ext_input=u_test, w_in=w_in,
        output_nodes=output_nodes
    ) 

    MC_train, MC_test, e_train, e_test = readout_module.run_task(
        X=(rs_train, rs_test), y=(y_train, y_test)
    )  # dataframe with score for each of the tau values

    # Interpolate the value where memory capacity becomes lower than 50% to get more precision
    return {'alpha': alpha, 'target': target, 'MC': MC_test}

# Main block to parallelize the simulation
if __name__ == "__main__":
    # Optional: parallel processing
    with Pool(processes=45) as pool:
        results = pool.map(run_simulation, params.values)

    # Write the results to a .csv file
    pd.DataFrame(results).to_csv(f"param_samples_{N}_results.csv")
