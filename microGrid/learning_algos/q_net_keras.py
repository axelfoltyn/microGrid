"""
Code for general deep Q-learning using Keras that can take as inputs scalars, vectors and matrices

.. Author: Vincent Francois-Lavet
"""

import numpy as np
from tensorflow.keras.optimizers import SGD,RMSprop
from tensorflow.keras import backend as K
from deer.base_classes import LearningAlgo as QNetwork
from .NN_keras import NN # Default Neural network used
import tensorflow as tf
from tensorflow.keras.utils import plot_model

class MyQNetwork(QNetwork):
    """
    Deep Q-learning network using Keras (with any backend)
    
    Parameters
    -----------
    environment : object from class Environment
        The environment in which the agent evolves.
    rho : float
        Parameter for rmsprop. Default : 0.9
    rms_epsilon : float
        Parameter for rmsprop. Default : 0.0001
    momentum : float
        Momentum for SGD. Default : 0
    clip_norm : float
        The gradient tensor will be clipped to a maximum L2 norm given by this value.
    freeze_interval : int
        Period during which the target network is freezed and after which the target network is updated. Default : 1000
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    update_rule: str
        {sgd,rmsprop}. Default : rmsprop
    random_state : numpy random number generator
    double_Q : bool, optional
        Activate or not the double_Q learning.
        More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
    neural_network : object, optional
        default is deer.learning_algos.NN_keras
    """

    def __init__(self, environment, rho=0.9, rms_epsilon=0.0001, momentum=0, clip_norm=1, freeze_interval=1000, batch_size=32, update_rule="rmsprop", random_state=np.random.RandomState(), double_Q=False, neural_network=NN):
        """ Initialize environment
        
        """
        QNetwork.__init__(self,environment, batch_size)

        #hyper param
        self._rho = rho
        self._rms_epsilon = rms_epsilon
        self._momentum = momentum
        self._clip_norm = clip_norm
        self._update_rule = update_rule
        self._freeze_interval = freeze_interval
        #self._double_Q = double_Q
        self._random_state = random_state
        self.update_counter = 0


        Q_net = neural_network(self._batch_size, self._input_dimensions, self._n_actions, self._random_state)
        self._dqn, self.params = Q_net._buildDQN()

        #plot_model(self._dqn, show_shapes=True)
        #self._compile()

        #self.q_vals, self.next_params = Q_net._buildDQN()
        #self.q_vals.compile(optimizer='rmsprop', loss='mse') #The parameters do not matter since training is done on self.q_vals

        self._resetQHat()

    def getAllParams(self):
        """ Get all parameters used by the learning algorithm

        Returns
        -------
        Values of the parameters: list of numpy arrays
        """
        params_value=[]
        for p in self.params:
            params_value.append(K.get_value(p))
        return params_value

    def setAllParams(self, list_of_values):
        """ Set all parameters used by the learning algorithm

        Arguments
        ---------
        list_of_values : list of numpy arrays
             list of the parameters to be set (same order than given by getAllParams()).
        """
        for i,p in enumerate(self.params):
            K.set_value(p,list_of_values[i])

    def flatten(self, x):
        result = []
        for el in x:
            if hasattr(el, "__iter__") and not isinstance(el, str):
                result.extend(self.flatten(el))
            else:
                result.append(el)
        return result

    def train(self, replay_memory, minibatch_size=32):
        """
        Train the Q-network from one batch of data.

        Parameters
        -----------
        states_val : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)].
        actions_val : numpy array of integers with size [self._batch_size]
            actions[i] is the action taken after having observed states[:][i].
        rewards_val : numpy array of floats with size [self._batch_size]
            rewards[i] is the reward obtained for taking actions[i-1].
        next_states_val : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)].
        terminals_val : numpy array of booleans with size [self._batch_size]
            terminals[i] is True if the transition leads to a terminal state and False otherwise

        Returns
        -------
        Average loss of the batch training (RMSE)
        Individual (square) losses for each tuple
        """
        # reset hyper parameter
        if self.update_counter % self._freeze_interval == 0:
            self._resetQHat()
        loss = []

        #Author : https://tomroth.com.au/dqn-simple/
        # and modified to work with this environment

        # choose <s,a,r,s',done> experiences randomly from the memory
        minibatch = np.random.choice(replay_memory, minibatch_size, replace=True)
        # create one list containing s, one list containing a, etc
        s_l = np.array(list(map(lambda x: x['s'], minibatch)))
        """for elem in s_l:
            for i in range(len(elem)):
                np.reshape(elem[i], tuple(list(self._dqn.inputs[i].shape)[1:]))
        print(self._dqn.inputs)
        print("s_l", s_l, s_l.shape)
        for seq in s_l: print(np.array(seq).shape)
        [print(i.shape, i.dtype) for i in self._dqn.inputs]"""
        a_l = np.array(list(map(lambda x: x['a'], minibatch)))
        r_l = np.array(list(map(lambda x: x['r'], minibatch)))
        sprime_l = np.array(np.array(list(map(lambda x: x['sprime'], minibatch))))

        """
        sprime_l = [[np.expand_dims(state, axis=0) for state in state_val] for state_val in sprime_l]
        sprime_l = []
        for elem in sprime_ltmp:
            print("ELEM", elem, sprime_l)
            elem_tmp = []
            for i in range(len(elem)):
                if not hasattr(elem[i], '__len__'):
                    elem_tmp.append(tf.convert_to_tensor(np.array([elem[i]], dtype=np.float32)))
                else:
                    elem_tmp.append(tf.convert_to_tensor(np.array(elem[i], dtype=np.float32)))
            sprime_l.append(elem_tmp)
            print("ELEMMM", elem_tmp, sprime_l)
            for elem in sprime_l:
            print("ELEM",elem)
            for i in range(len(elem)):
                if not hasattr(elem[i], '__len__'):
                    elem[i] = np.asarray(np.array(elem[i]), dtype=np.float32)
                else:
                    np.reshape(elem[i], tuple([-1] + list(self._dqn.inputs[i].shape)[1:]))
            print("ELEMMM", elem)"""
        #sprime_l = np.array(sprime_l)
        #sprime_l = tf.convert_to_tensor(sprime_ltmp)
        """print(sprime_l, type(sprime_l))
        print("sprime_l", sprime_l, sprime_l.shape)
        for seq in sprime_l: print(np.array(seq).shape)
        [print(i.shape, i.dtype) for i in self._dqn.inputs]"""
        #sprime_l = self.flatten(sprime_l)


        done_l = np.array(list(map(lambda x: x['done'], minibatch)))
        # Find q(s', a') for all possible actions a'. Store in list
        # We'll use the maximum of these values for q-update
        #sprime_l = [sprime_l[:,i] for i in range(len(sprime_l[0]))]
        #print("???",[tf.expand_dims(elem, -1) for elem in sprime_l])
        qvals_sprime_l = self._dqn.predict(sprime_l)
        # Find q(s,a) for all possible actions a. Store in list
        target_f = self._dqn.predict(s_l)
        # q-update target
        # For the action we took, use the q-update value
        # For other actions, use the current nnet predicted value
        for i, (s, a, r, qvals_sprime, done) in enumerate(zip(s_l, a_l, r_l, qvals_sprime_l, done_l)):
            if not done:
                target = r + self._df * np.max(qvals_sprime)
            else:
                target = r
            loss.append(target - qvals_sprime)
            target_f[i][a] = target
        # Update weights of neural network with fit()
        # Loss function is 0 for actions we didn't take
        self._dqn.fit(s_l, target_f, epochs=1, verbose=0)
        return np.mean(loss), loss


    def qValues(self, state_val):
        """ Get the q values for one belief state

        Arguments
        ---------
        state_val : one belief state

        Returns
        -------
        The q values for the provided belief state
        """
        state_val = tf.expand_dims(state_val, axis=0)
        return self._dqn.predict(state_val)[0]
        #return self._dqn.predict([np.expand_dims(state, axis=0) for state in state_val])[0]

    def chooseBestAction(self, state, *args, **kwargs):
        """ Get the best action for a pseudo-state

        Arguments
        ---------
        state : one pseudo-state

        Returns
        -------
        The best action : int
        """        
        q_vals = self.qValues(state)

        return np.argmax(q_vals),np.max(q_vals)
        
    def _compile(self):
        """ Compile self.q_vals
        """
        
        if (self._update_rule=="sgd"):
            optimizer = SGD(lr=self._lr, momentum=self._momentum, nesterov=False, clipnorm=self._clip_norm)
        elif (self._update_rule=="rmsprop"):
            optimizer = RMSprop(lr=self._lr, rho=self._rho, epsilon=self._rms_epsilon, clipnorm=self._clip_norm)
        else:
            raise Exception('The update_rule '+self._update_rule+' is not implemented.')
        
        self._dqn.compile(optimizer=optimizer, loss='mse')

    def _resetQHat(self):
        """ Set the target Q-network weights equal to the main Q-network weights
        """
        
        """for (param,next_param) in zip(self.params, self.next_params):
            K.set_value(next_param,K.get_value(param))"""

        self._compile() # recompile to take into account new optimizer parameters that may have changed since
                        # self._compile() was called in __init__. FIXME: this call should ideally be done elsewhere
