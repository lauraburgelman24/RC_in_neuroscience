import numpy as np
import pandas as pd
import random

import os
import sys

from conn2res.connectivity import Conn
import RC_lesion as RC
import task_class as tc
from conn2res import readout

# Set the necessary parameters
INPUT_GAIN = 1

# Generate task data
task = tc.MemoryCapacity(n = 6000, tau=20) # Use the updated memory capacity task that was proposed in this project
u_train, u_test, y_train, y_test = task.fetch_data(input_gain=INPUT_GAIN, train=0.8)

# Save task data
np.savez(f"tasks/memory_task.npz", u_train=u_train, u_test=u_test, y_train=y_train, y_test=y_test)