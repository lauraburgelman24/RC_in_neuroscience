import numpy as np
import random
from scipy.linalg import eigh
import pandas as pd

from conn2res.connectivity import Conn
from conn2res.tasks import Task
from conn2res.reservoir import EchoStateNetwork
from conn2res.readout import Readout, train_test_split, _check_xy_type, _check_x_dims
from conn2res import plotting

# Connectivity matrix definition:
# We want to be able to build from scratch (without human representation) as well (by defining nodes, sparsity, distribution weights)
# We want to use W_link as self.w - a binarised version of the matrix that represents the neighbours - this is constant whereas w itself is not

class ConnPlasticity(Conn):
    """ 
    Class inheriting from conn2res Conn adding the option to construct a randomised w and defining W_link as binarised version of w.
    This extra method of constructing the connectome was inspired by the work of Falandays et al. (2024).

    Parameters
    ----------
    w: numpy.ndarray, optional
        the (weighted or binary) connectivity matrix
    filename: str, optional
        the filename of the file (.npy) where the connectivity matrix is stored
    subj_id: int, optional
        the subject id of the connectivity matrix in the case that is 3D
    N: int, optional
        the number of nodes for the connectivity matrix to construct, default 200
    rho: float, optional
        the sparseness of the connectivity matrix to construct, default .9
    dist: str, optional
        the name of the distribution of the weights for the connectivity matrix to construct, default "normal"
        other option "uniform"

    Raises
    ------
    ValueError
        if dist contains another str than "normal" or "uniform"
    """

    def __init__(self, w=None, filename=None, subj_id=0, N = 200, rho = 0.9, dist = 'normal', **kwargs):
        if (w is None) and (filename is None):
            W_link = np.zeros((N, N)) # matrix indicating the neighbours
            for row in range(W_link.shape[0]):
                for col in range(W_link.shape[1]):
                    if row == col:
                        continue # diagonal elements are zero
                    W_link[row, col] = random.choices([0, 1], weights = (rho, 1-rho), k=1)[0] # rho indicates sparseness
            
            w = np.zeros((N, N))
            for row in range(w.shape[0]):
                for col in range(w.shape[1]):
                    if W_link[row, col] == 1:
                        if dist.lower() == "normal":
                            w[row, col] = np.random.normal(0, 1)
                        elif dist.lower() == "uniform":
                            w[row, col] = np.random.uniform(low=-1, high=1, size=1)[0]
                        else:
                            raise ValueError("dist must be either 'normal' of 'uniform'")
            super().__init__(w=W_link, **kwargs)
            self.w_start = w
        else:
            super().__init__(w=w, filename=filename, subj_id=subj_id, **kwargs)
            self.w_start = self.w.copy()
            self.binarize() # making a linking matrix out of w

    def get_input_matrix(self, N = 1, input_nodes = None, rho = 0.9, weight = 0.75):
        #TODO: also allow normal / uniform distribution in weights
        """
        Function that returns the input matrix. Either based on node locations, name of brain subsection or sparseness of input.

        Parameters
        ----------
        N: int, optional
            the number of inputs given to the reservoir, default 1
        input_nodes: str or np.ndarray, optional
            the name of the brain section or the locations of the nodes
        rho: float, optional
            the sparseness of the input matrix, 1 - prob(connection), default 0.9
        weight: float, optional
            the weight of the input connections, default .75
        """
        w_in = np.zeros((N, self.n_nodes))

        if isinstance(input_nodes, str):
            input_nodes = self.get_nodes(nodes_from=None, node_set=input_nodes)
            w_in[:, input_nodes] = np.eye(1)
            return w_in
        elif isinstance(input_nodes, np.ndarray):
            w_in[:, input_nodes] = np.eye(1)
            return w_in

        for row in range(w_in.shape[0]):
            for col in range(w_in.shape[1]):
                    w_in[row, col] = random.choices([0, weight], weights = (rho, 1-rho), k=1)[0]
        return w_in

    def get_output_matrix(self, N = 1, output_nodes = None, rho = 0.9, weight = 1):
        #TODO: also allow normal / uniform distribution in weights
        """
        Function that returns the output matrix. Either based on node locations, name of brain subsection or sparseness of input.

        Parameters
        ----------
        N: int, optional
            the number of outputs given to the reservoir, default 1
        output_nodes: str or np.ndarray, optional
            the name of the brain section or the locations of the nodes
        rho: float, optional
            the sparseness of the output matrix, 1 - prob(connection), default 0.9
        weight: float, optional
            the weight of the output connections, default 1
        """
        w_out = np.zeros((self.n_nodes, N))

        if isinstance(output_nodes, str):
            output_nodes = self.get_nodes(nodes_from=None, node_set=output_nodes)
            w_out[output_nodes, :] = np.eye(1)
            return w_out
        elif isinstance(output_nodes, np.ndarray):
            w_out[output_nodes, :] = np.eye(1)
            return w_in

        for row in range(w_out.shape[0]):
            for col in range(w_out.shape[1]):
                    w_out[row, col] = random.choices([0, weight], weights = (rho, 1-rho), k=1)[0]
        return w_out

# Task definition
#TODO:
#make task interactive such that output has an impact on the input
#currently implemented during readout

class RotationTask(Task):
    """
    This class provides the rotating signal that was described in Falandays et al. (2024).
    No target signal y is provided.

    Parameters
    ----------
    name: str
        "Rotation" - this class does not provide other tasks
    n_trials: int, optional
        the number of possible rotations, default 10
    n: int, optional
        the length of 1 trial
    """
    def __init__(self, name, n_trials=10, n=720):
        super().__init__(name, n_trials)
        self.n = n

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name.lower() != "rotation":
            raise ValueError("Only 'Rotation' is provided")
        self._name = name

    def fetch_data(self, n_trials=None, initial_position = 0):
        """
        Function that provides the rotating signal, the signal has a chance of 50% of changing direction after every trial of length n.
        This is an adaptation of the original rotation task presented in Falandays et al. (2024).

        Parameters
        ----------
        n_trials: int, optional
            the number of full rotations the robot needs to perform, default 10
        initial_position: int, optional
            the starting position of the robot, default 0
        """
        if n_trials is not None:
            self.n_trials = n_trials

        u = [] # list for the rotation signal

        direction = 1

        for n in range(self.n_trials):
            direction = random.choices([-1, 1], weights = (0.5, 0.5), k=1)[0]
            u += [(i*direction)%360 for i in range(self.n)]
        return np.array(u)

class PlasticityReservoir():
    """
    This class integrates the work of Suárez et al. (2024) with that of Falandays et al. (2024).
    As a result, the reservoir built with this function has both synaptic and internal plasticity.
    Plasticity can be turned "on" and "off".

    Parameters
    ----------
    W: np.ndarray
        Binary version of the connectivity matrix indicating the links between neighbours, shape N x N
    W: np.ndarray
        The connectivity matrix of shape N x N
    W_in: np.ndarray
        The input matrix of shape I x N where I is the number of inputs that go into the model
    plasticity: boolean, optional
        Indicates whether plasticity is turned on or off, default False
    W_out: np.ndarray, optional
        The output matrix is only necessary when plasticity = True, default None
    """
    def __init__(self, W_link, W, W_in, plasticity = False, W_out = None):
        self.W_link = W_link.copy()
        self.W = W.copy()
        self.W_in = W_in.copy()
        if (plasticity is True) and (W_out is None):
            raise ValueError("W_out should be defined when using plasticity")
        self.plasticity = plasticity
        if W_out is not None:
            self.W_out = W_out.copy()

    def run_task(self, U=None, f=None, y=None, **kwargs):
        """
        Function that calls "run_plasticity" if plasticity is True, otherwise "run_no_plasticity"

        "run_plasticity" takes arguments: U, f, T_min, output, leaking, lr_W, lr_T, Amp_out, task and additional arguments for f
        "run_no_plasticity" takes arguments: U, y, activation_function, ALPHAS, output_nodes, METRIC
        """
        if self.plasticity is True:
            if U is None or f is None:
                raise ValueError("Both input U and input function f should be specified")
            outputs, s_values, T_values = self.run_plasticity(U, f, **kwargs)
            return outputs, s_values, T_values
        else:
            df_alpha = self.run_no_plasticity(U, y, **kwargs)
            return df_alpha

    def run_plasticity(self, U, f, T_min = 1, output = 90, leaking = .75, lr_W = 1, lr_T = .01, Amp_out = 10, task = 'robot', **kwargs):
        """
        Run the task when plasticity is implemented.

        Parameters
        ----------
        U: np.ndarray
            The signal that needs to be traced. The input to the model is derived from this.
        f: function
            The function used to get the actual input to the model. The function should take at least 2 arguments: current u & previous output
            additional parameters can be fed to the function
        T_min: int or float, optional
            The minimum target value for the nodes, default 1
        output: int or float, optional
            The initialisation of the output, default 90
        leaking: float, optional
            The fraction of the node value that remains after one time step, default .75
        lr_W: float, optional
            The learning rate for connectivity matrix W, default 1
        lr_T: float, optional
            The learning rate for target level T, default .01
        Amp_out: float, optional
            The amplification of the output, default 10
        task: str, optional
            Determines the way in which the output is generated. Currently only built for 'robot'. Default 'robot'
        """
        self.s = np.zeros(self.W.shape[0])
        self.x = np.zeros(self.W.shape[0])
        self.T = np.ones(self.W.shape[0])
        self.leaking = leaking
        self.lr_W = lr_W
        self.lr_T = lr_T

        outputs = [output]
        s_values = [self.s]
        T_values = [self.T]
        for u in U:
            i = f(u, output, **kwargs) #the actual input to the model

            prev_s = self.s.copy()

            E = self.get_activation(i) # this also updates self.x and self.s

            self.learning(prev_s, E, T_min)

            output = self.get_output(task, output, Amp_out)
            outputs.append(output)
            s_values.append(self.s)
            T_values.append(self.T)
        return np.array(outputs), np.array(s_values), np.array(T_values)

    def get_activation(self, i):
        """
        This function is fully based on the work of Falandays et al. (2024).
        """
        x = self.x * self.leaking + i @ self.W_in + self.s @ self.W

        T_thresh = 2 * self.T

        # TODO: Look into alternative activation functions
        s = self.s.copy()
        s[x >= T_thresh] = 1
        s[x < T_thresh] = 0
        self.s = s

        x[s == 1] = x[s == 1] - T_thresh[s == 1] 
        x[x < 0] = 0
        self.x = x

        E = self.x - self.T # error is NEW x - original target

        return E

    def learning(self, prev_s, E, T_min):
        """
        This learning function is fully based on the work of Falandays et al. (2024).
        """
        # which nodes are inactive
        inactive = np.argwhere(prev_s <= 0)[:, 0]
        
        # find the number of active neighbours for each node
        active_neighbours = self.W_link.copy()
        active_neighbours[inactive, :] = 0
        active_neighbours = np.sum(active_neighbours, axis = 0) #1 row with number of active neighbours

        # update W
        update = np.zeros((self.W.shape[0], self.W.shape[1]))
        update[:, :] = E * self.lr_W #error times learning rate
        update[self.W_link == 0] = 0 #no connection
        update[inactive, :] = 0 #no spiking
        update = update / active_neighbours #element-wise division
        update = np.nan_to_num(update) # could be division by zero, then "update" should be zero
        self.W -= update # update matrix W

        # update targets
        T = self.T.copy()
        T += E * self.lr_T
        T[T < T_min] = T_min # targets cannot go below 1
        self.T = T

    def get_output(self, task, output, Amp_out):
        if task.lower() == 'robot':
            output_acts = np.dot(self.s, self.W_out)/np.sum(self.W_out, axis=0)
            diff = (output_acts[0] - output_acts[1]) * Amp_out
            output = (output + diff)%360
        return output

    def run_no_plasticity(self, U, y, activation_function = 'tanh', ALPHAS = [1], output_nodes = None, METRIC = ['corrcoef'], alpha_vis = [0.8], taskname = "task"):
        """
        Function that runs the task when no plasticity is present. The function follows the workflow described in the Conn2res package.

        Parameters
        ----------
        U: np.ndarray
            The input values to the matrix
        y: np.ndarray
            The target output values
        activation_function: str, optional
            {'linear', 'elu', 'relu', 'leaky_relu','sigmoid', 'tanh', 'step'} (nonlinearity of the system's units), default 'tanh'
        ALPHAS: list, optional
            list of alpha values for which the model should be calculated, default [1]
        output_nodes : list or numpy.ndarray, optional
            List of nodes for which reservoir states will be returned if 'return_states' is True.
        METRIC: list, optional
            the metrics that should be calculated for this test, default ['corrcoef']
        alpha_vis: list, optional
            list of alpha values for which you want visualisation
        taskname: str, optional
            name of the task for plot title

        Returns
        -------
        df_alpha: pd.dataframe
            Dataframe representing the metrics for each of the alpha values
        """
        # make sure the spectral radius is 1
        self.scale_and_normalise()

        # define the network
        esn = EchoStateNetwork(w=self.W, activation_function= activation_function)

        if np.ndim(U) == 1:
            U = U.reshape(-1, 1)

        #define the readout module, this also defines the estimator
        readout_module = Readout_y(y=y)

        #the order is preserved with this split! It takse the first 70% as train, the last 30% as test
        x_train, x_test, y_train, y_test = train_test_split(U, y)

        # Create a DataFrame with dynamic column names and row labels
        columns = [f"alpha = {a}" for a in ALPHAS]
        index = [f"{metric}" for metric in METRIC]

        # Initialize the DataFrame with these columns and index
        df_alpha = pd.DataFrame(index=index, columns=columns)

        for alpha in ALPHAS:
            esn.w = alpha * self.W

            rs_train = esn.simulate(ext_input=x_train, w_in=self.W_in, output_nodes=output_nodes)

            rs_test = esn.simulate(ext_input=x_test, w_in=self.W_in, output_nodes=output_nodes)

            df_res = readout_module.run_task(X=(rs_train, rs_test), y=(y_train, y_test), metric=METRIC)

            if alpha in alpha_vis:
                plotting.plot_diagnostics(
                        x=x_test, y=y_test, reservoir_states=rs_test,
                        trained_model=readout_module.model, title=f'{taskname} - test',
                        rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
                        show=True
                        )
            
            a = np.round(alpha, 3)

            for metric in METRIC:
                df_alpha.at[metric, f"alpha = {a}"] = df_res[metric].iloc[0]
        return df_alpha

    def scale_and_normalise(self):
        """
        Scale the connectivity matrix between [0, 1] - then normalise by spectral radius
        """
        # scale connectivity matrix between [0, 1]
        self.W = (self.W - self.W.min()) / (self.W.max() - self.W.min())

        ew, _ = eigh(self.W)

class Readout_y(Readout):
    """
    Class inheriting from Readout in conn2res that also provides the prediction of the model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_prediction(self, X, y):

        # check X and y are arrays
        X, y = _check_xy_type(X, y)

        # check X and y dimensions
        X = _check_x_dims(X)

        return self._model.predict(X)