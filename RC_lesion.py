import numpy as np
import random
from scipy.linalg import eigh
import pandas as pd
import matplotlib.pyplot as plt

from conn2res.connectivity import Conn
from conn2res.tasks import Task
from conn2res.reservoir import EchoStateNetwork
from conn2res.readout import Readout, train_test_split, _check_xy_type, _check_x_dims
from conn2res import plotting

class ESNLesion(EchoStateNetwork):
    """
    Class that represents an Echo State Network

    """

    def __init__(self, w, *args, activation_function='tanh', targets = None, leak_rate = None, n_lesions = None, input_nodes = None, output_nodes = None, plasticity = False, lr = 0.01, window = 30, delay = False, coords = None, dtv = 0.02, **kwargs):
                 
        """
        Constructor class for Echo State Networks. This code is compatible with the Conn2res toolbox (Suárez et al., 2024). Plasticity is based on the principle of Falandays et al. (2024). Conduction delays are based on Iacob and Dambre (2024).

        Parameters
        ----------
        w: (N, N) numpy.ndarray
            Reservoir connectivity matrix (source, target)
            N: number of nodes in the network. If w is directed, then rows
            (columns) should correspond to source (target) nodes.
        activation_function: str {'linear', 'elu', 'relu', 'leaky_relu',
            'sigmoid', 'tanh', 'step'}, default 'tanh'
            Activation function (nonlinearity of the system's units)
        targets: (N,) numpy.ndarray, default None
            The target values used in the activation function, these are updated when plasticity = True
        leak_rate: float in (0, 1], default None
            A leaky integrator is used if leak_rate is not None.
            The leak rate forms a time constant to control the dynamics.
        n_lesions: int or list, default None
            The number of lesioned nodes in the network or a list representing the lesioned nodes. These nodes' state will remain zero.
        input_nodes: (n, u) numpy.ndarray, default None
            The n nodes that receive input signal u
        output_nodes: (n,) numpy.ndarray, default None
            The n nodes that provide the readout signals
        plasticity: Boolean, default False
            Indicates whether plasticity, so updating of the target, is turned on or off
        lr: float, default 0.01
            Learning rate, indicates the step size of the plasticity updates
        window: int, default 30
            The number of previous values that are taken into account when calculating the plasticity update
        delay: Boolean, default False
            Indicates whether distance based delays are used in the model
        coords: (N, 3) numpy.ndarray, default None
            Coordinates of the reservoir nodes in 3 dimensions
        dtv: float, default 0.02
            Multiplication of the smallest time step and the conduction speed
        """

        super().__init__(w, *args, **kwargs)

        # lesioning
        self.n_lesions = n_lesions

        # generate mask for lesioning
        mask = np.ones(self.w.shape[0])
        if self.n_lesions is not None:
            if isinstance(self.n_lesions, list):
                mask[self.n_lesions] = 0
            else:
                idx = np.arange(self.w.shape[0])
                # 1. identify the input nodes
                idx_in = input_nodes
                # 2. identify the output nodes
                idx_out = output_nodes
                # 3. randomly lesion nodes other than input or output
                idx_lesion = list(set(idx) - set(idx_in) - set(idx_out))
                lesions = random.sample(idx_lesion, self.n_lesions)
                mask[lesions] = 0
        self.mask = mask

        # Reservoir parameters
        self.coords = coords
        self.N = self.w.shape[0]
        self.leak_rate = leak_rate
        self.activation_function = self.set_activation_function(
            activation_function)
        
        # Learning parameters
        if targets is None:
            self.T = np.repeat(0.0, self.w.shape[0])
        else:
            self.T = targets
        self.plasticity = plasticity
        if plasticity is True:
            self.lr = lr
            self.window = np.zeros((self.N, window))

        # See global changes
        self.delta = []

        # Delays
        self.delay = delay
        if delay is True:

            # Euclidean distance matrix S
            self.S = coordinates2distance(coords)

            # Determine discrete distance matrix
            self.dtv = dtv
            self.D = np.asarray(np.floor(self.S / dtv), dtype='int32')

            # Max possible distance with buffer
            self.D_max = np.max(self.D) + 1
            print(self.D_max)

            # Storage for the calculation of y in A
            self.A = np.zeros((self.N, self.D_max))

            # Compute the D_max weight matrices that indicate the delays
            self.W_masked_list = [self.w]
            if not (self.w is None) and (self.D_max > 2):
                self.compute_masked_W()
            
            self.W_masked_list_init = [np.copy(partial_W) for partial_W in self.W_masked_list]

    def set_activation_function(self, function):
        def linear(x, m = 0.5):
            s = m * x - 2 * self.T
            if self.plasticity is True:
                self.update_T(s)
            return s

        def step(x, vmin = 0, vmax = 1):
            s = np.piecewise(x, [x < 2 * self.T, x >= 2 * self.T], [vmin, vmax]).astype(int)
            if self.plasticity is True:
                self.update_T(s)
            return s

        def tanh(x):
            s = np.tanh(x - 2 * self.T)
            if self.plasticity is True:
                self.update_T(s)
            return s

        def sigmoid(x):
            s = 1.0 / (1 + np.exp(-x + 2 * self.T))
            if self.plasticity is True:
                self.update_T(s)
            return s

        def relu(x):
            s = np.maximum(0, x - 2 * self.T)
            if self.plasticity is True:
                self.update_T(s)
            return s

        def elu(x, alpha = 0.5):
            s = x.copy()
            s[s <= self.T] = alpha * (np.exp(s[s <= 2 * self.T]) - 1)
            if self.plasticity is True:
                self.update_T(s)
            return s

        def leaky_relu(x, alpha = 0.5):
            s = np.maximum(alpha * x, x - 2 * self.T)
            if self.plasticity is True:
                self.update_T(s)
            return s
            
        if function == 'linear':
            return linear
        elif function == 'elu':
            return elu
        elif function == 'relu':
            return relu
        elif function == 'leaky_relu':
            return leaky_relu
        elif function == 'sigmoid':
            return sigmoid
        elif function == 'tanh':
            return tanh
        elif function == 'step':
            return step

    def update_T(self, s):
        # Thresholds before
        d1 = self.T.copy()

        # Update values in window
        self.window[:, 1:] = self.window[:, :-1]
        self.window[:, 0] = s.copy()

        # Update thresholds
        avg = np.mean(self.window, axis=1)
        E = avg - self.T
        self.T += self.lr * E
        self.T[self.T > 1] = 1
        self.T[self.T < -1] = -1

        # Thresholds after
        d2 = self.T.copy()
        self.delta.append(np.sum(np.abs(d1-d2)))

    def compute_masked_W(self):
        """
        Creates a list of masked weight matrices, i.e. weight matrices containing only the weight of connections with a specified delay.
        Returns: None
        """
        self.W_masked_list.clear()
        for step in range(self.D_max):
            # Create mask for each step
            mask = self.D == step
            # Elementwise product with buffer mask to only add connectivity for specific delay
            buffW = np.multiply(mask, self.w)
            self.W_masked_list.append(buffW)
    

    def simulate(
        self, ext_input, w_in, ic=None, output_nodes=None,
        return_states=True, **kwargs
    ):
        """
        Simulates reservoir dynamics given an external input signal
        'ext_input' and an input connectivity matrix 'w_in'

        Parameters
        ----------
        ext_input : (time, N_inputs) numpy.ndarray
            External input signal
            N_inputs: number of external input signals
        w_in : (N_inputs, N) numpy.ndarray
            Input connectivity matrix (source, target)
            N_inputs: number of external input signals
            N: number of nodes in the network
        ic : (N,) numpy.ndarray, optional
            Initial conditions
            N: number of nodes in the network. If w is directed, then rows
            (columns) should correspond to source (target) nodes.
        output_nodes : list or numpy.ndarray, optional
            List of nodes for which reservoir states will be returned if
            'return_states' is True.
        return_states : bool, optional
            If True, simulated resrvoir states are returned. True by default.
        kwargs:
            Other keyword arguments are passed to self.activation_function

        Returns
        -------
        self._state : (time, N) numpy.ndarray
            Activation states of the reservoir.
            N: number of nodes in the network if output_nodes is None, else
            number of output_nodes
        """

        # if ext_input is list or tuple convert to numpy.ndarray
        if isinstance(ext_input, (list, tuple)):
            sections = utils.get_sections(ext_input)
            ext_input = utils.concat(ext_input)
            convert_to_list = True
        else:
            convert_to_list = False

        # initialize reservoir states
        timesteps = range(1, len(ext_input) + 1)
        self._state = np.zeros((len(timesteps) + 1, self.n_nodes))

        # set initial conditions
        if ic is not None:
            self._state[0, :] = ic * self.mask # make sure that lesioned nodes have state 0

        # simulate dynamics
        for t in timesteps:
            if self.delay is True:
                # Remove the last column to make room for the new state values
                self.A[:, 1:] = self.A[:, :-1]

                # Current implementation: no delays in the input values
                y = np.dot(ext_input[t-1, :], w_in)
                if self.D_max > 2:
                    for d, masked_weights in enumerate(self.W_masked_list):
                        y += np.matmul(masked_weights, self.A[:, d])
                else:
                    y += np.matmul(self.w, self.A[:, 0])

                y = self.activation_function(y)

                if self.leak_rate is None:
                    self.A[:, 0] = y * self.mask
                else:
                    self.A[:, 0] = ((1 - self.leak_rate) * self.A[:, 0] + self.leak_rate * y) * self.mask
                self._state[t, :] = np.copy(self.A[:, 0])

            else:
                if self.leak_rate is None:
                    synap_input = np.dot(
                        self._state[t-1, :], self.w) + np.dot(ext_input[t-1, :], w_in)
                    self._state[t, :] = self.activation_function(synap_input, **kwargs) * self.mask
                else:
                    synap_input = np.dot(
                        self._state[t-1, :], self.w) + np.dot(ext_input[t-1, :], w_in)
                    self._state[t, :] = ((1.0-self.leak_rate)*self._state[t-1,:] + self.leak_rate*self.activation_function(synap_input, **kwargs)) * self.mask
    
        # remove initial condition (to match the time index of _state and ext_input)
        self._state = self._state[1:]

        # convert back to list or tuple
        if convert_to_list:
            self._state = utils.split(self._state, sections)

        # return the same type
        if return_states:
            if output_nodes is not None:
                if convert_to_list:
                    return [state[:, output_nodes] for state in self._state]
                else:
                    return self._state[:, output_nodes]
            else:
                return self._state


def coordinates2distance(coords):
    """
    This function is adapted from 2D to 3D from Iacob, S. & Dambre, J.: https://github.com/StefanIacob/DDN-public/blob/main/network.py
    The function transforms the neurons coordinates to a distance matrix
    """
    N = coords.shape[0]
    S = np.zeros((N, N))

    def dist(dist_x, dist_y, dist_z):
        return np.sqrt(dist_x **2 + dist_y**2 + dist_z**2)

    for i in range(N):
        for j in range(N):
            if not i == j:
                dist_x = np.abs(coords[i, 0] - coords[j, 0])
                dist_y = np.abs(coords[i, 1] - coords[j, 1])
                dist_z = np.abs(coords[i, 2] - coords[j, 2])
                s = dist(dist_x, dist_y, dist_z)
                S[i, j] = s
    return S
