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
import torch
import gc
# Add to leaky code within python_script_being_profiled.py
from pympler import muppy, summary
import pandas as pd


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
        default is deerv2.learning_algos.NN_keras
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
        print("create model")
        self._dqn, self.params = Q_net._buildDQN()

        self._compile()

        self._dqn_fix, self.next_params = Q_net._buildDQN()
        self._dqn_fix.compile(optimizer='rmsprop', loss='mse') #The parameters do not matter since training is done on self.q_vals

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
        #Author : https://tomroth.com.au/dqn-simple/
        # and modified to work with this environment

        # choose <s,a,r,s',done> experiences randomly from the memory
        minibatch = np.random.choice(replay_memory, minibatch_size, replace=True)
        # create one list containing s, one list containing a, etc
        s_l_tmp = np.array(list(map(lambda x: x['s'], minibatch)), dtype='object') #, dtype='object' to remove warnning
        s_l = [[] for _ in range(len(s_l_tmp[0]))]
        for elem_l in s_l_tmp:
            for i, elem in enumerate(elem_l):
                if hasattr(elem, '__len__'):
                    s_l[i].append(elem)
                else:
                    s_l[i].append([elem])
        s_l = [np.array(elem, dtype=np.float32) for elem in s_l]
        a_l = np.array(list(map(lambda x: x['a'], minibatch)))
        r_l = np.array(list(map(lambda x: x['r'], minibatch)))
        sprime_l_tmp = np.array(list(map(lambda x: x['sprime'], minibatch)), dtype='object')
        sprime_l = [[] for _ in range(len(sprime_l_tmp[0]))]
        for elem_l in sprime_l_tmp:
            for i, elem in enumerate(elem_l):
                if hasattr(elem, '__len__'):
                    sprime_l[i].append(elem)
                else:
                    sprime_l[i].append([elem])
        sprime_l = [np.array(elem, dtype=np.float32) for elem in sprime_l]
        done_l = np.array(list(map(lambda x: x['done'], minibatch)))
        qvals_sprime_l = self._dqn_fix.predict(sprime_l)
        # Find q(s,a) for all possible actions a. Store in list
        q_table = self._dqn.predict(s_l)
        # q-update target
        # For the action we took, use the q-update value
        # For other actions, use the current nnet predicted value
        max_next_q_vals = np.max(qvals_sprime_l, axis=1, keepdims=True)
        not_terminals = np.invert(done_l).astype(float)
        target = r_l + not_terminals * self._df * max_next_q_vals.reshape((-1))
        q_val = q_table[np.arange(self._batch_size), a_l]
        diff = - q_val + target
        loss_ind = pow(diff, 2)

        q_table[np.arange(self._batch_size), a_l] = target
        # Update weights of neural network with fit()
        # Loss function is 0 for actions we didn't take
        try:
            #print(len(gc.get_objects()))
            print("AVANT")
            all_objects = muppy.get_objects()
            sum1 = summary.summarize(all_objects)
            # Prints out a summary of the large objects
            summary.print_(sum1)
            # Get references to certain types of objects such as Tensor
            dataframes = [ao for ao in all_objects if isinstance(ao, tf.Tensor)]
            for d in dataframes:
                print(d.columns.values)
                print(len(d))
            print(len(gc.get_objects()))
            self._dqn.fit(s_l, q_table, verbose=0, use_multiprocessing = True)
            all_objects = muppy.get_objects()
            sum1 = summary.summarize(all_objects)
            # Prints out a summary of the large objects
            summary.print_(sum1)
            # Get references to certain types of objects such as Tensor
            dataframes = [ao for ao in all_objects if isinstance(ao, tf.Tensor)]
            for d in dataframes:
                print(d.columns.values)
                print(len(d))
            # loss = self._dqn.train_on_batch(s_l, q_table)
            del(q_table)
            del(qvals_sprime_l)
            tf.keras.backend.clear_session()
            gc.collect()
            print(len(gc.get_objects()))
            print("APRES")

        except Exception as e:
            print("s_l", s_l)
            print("q_table", q_table)
            print(len(gc.get_objects()))
            print(self._dqn.summary())
            raise e
        torch.cuda.empty_cache()
        return np.mean(loss_ind)/int(minibatch_size/32), loss_ind


    def qValues(self, state_val):
        """ Get the q values for one belief state

        Arguments
        ---------
        state_val : one belief state

        Returns
        -------
        The q values for the provided belief state
        """
        """sprime_l = [[] for _ in range(len(state_val[0]))]
        for k, elem_l in enumerate(state_val):
            for i, elem in enumerate(elem_l):
                if hasattr(elem, '__len__'):
                    sprime_l[i].append(elem)
                else:
                    sprime_l[i].append([elem])
        state_val = [np.array(elem, dtype=np.float32) for elem in sprime_l]"""
        #state_val = tf.expand_dims(state_val, axis=0)
        #return self._dqn.predict(state_val)[0]
        return self._dqn.predict([np.expand_dims(state, axis=0) for state in state_val])[0]

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
        action, val = np.argmax(q_vals),np.max(q_vals)
        del(q_vals)
        gc.collect()
        tf.keras.backend.clear_session()
        return action, val
        
    def _compile(self):
        """ Compile self.q_vals
        """
        #tf.keras.backend.clear_session()
        if (self._update_rule=="sgd"):
            optimizer = SGD(lr=self._lr, momentum=self._momentum, nesterov=False, clipnorm=self._clip_norm)
        elif (self._update_rule=="rmsprop"):
            optimizer = RMSprop(lr=self._lr, rho=self._rho, epsilon=self._rms_epsilon, clipnorm=self._clip_norm)
        else:
            raise Exception('The update_rule '+self._update_rule+' is not implemented.')
        #self._dqn.summary()
        self._dqn.compile(optimizer=optimizer, loss='mse')

    def _resetQHat(self):
        """ Set the target Q-network weights equal to the main Q-network weights
        """
        
        """for (param,next_param) in zip(self.params, self.next_params):
            K.set_value(next_param,K.get_value(param))"""

        self._compile() # recompile to take into account new optimizer parameters that may have changed since
                        # self._compile() was called in __init__. FIXME: this call should ideally be done elsewhere
