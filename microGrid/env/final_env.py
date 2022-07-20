"""
The environment simulates a microgrid consisting of short and long term storage. The agent can either choose to store in the long term storage or take energy out of it while the short term storage handle at best the lack or surplus of energy by discharging itself or charging itself respectively. Whenever the short term storage is empty and cannot handle the net demand a penalty (negative reward) is obtained equal to the value of loss load set to 2euro/kWh.
Two actions are possible for the agent:
- Action 0 corresponds to discharging the long-term storage
- Action 1 corresponds to charging the long-term storage
The state of the agent is made up of an history of two to four punctual observations:
- Charging state of the short term storage (0 is empty, 1 is full)
- Production and consumption (0 is no production or consumption, 1 is maximal production or consumption)
( - Distance to equinox )
( - Predictions of future production : average of the production for the next 24 hours and 48 hours )
More information can be found in the paper to be published :
Efficient decision making in stochastic micro-grids using deep reinforcement learning, Vincent Francois-Lavet, David Taralla, Raphael Fonteneau, Damien Ernst

modified by Axel Foltyn
"""
import gym
import numpy as np
import copy
from gym.spaces import Discrete, Box

import os

class MyEnv(gym.Env):
    def __init__(self, rng, reduce_qty_data=None, length_history=None, start_history=None,
                 consumption=None, production=None, scale_cons = 2.1, scale_prod = 12000./1000./2,
                 pred = False, dist_equinox = False, max_ener_buy=None, max_ener_sell=None, verbose=False):
        """ Initialize environment

        Arguments:
            rng - the numpy random number generator
            reduce_qty_data - ???
            length_history - size of the consumption and production history
            start_history - ???
            consumption - consumption data (normed value)
            production - production data (normed value)
            scale_cons - scale for consumption data
            scale_prod - scale for production data
            pred - if we want the future consumption and production in the observations
            dist_equinox - if we want the delay before the next summer solstice
            max_ener_buy - max energy we can take outside
            max_ener_sell - max energy we can give outside
        """

        self.save_state = []

        # dict_reward stores the reward function
        # dict_coeff stores the reward coefficient
        # (used in add_reward and my_reward)
        self.dict_reward = dict()
        self.dict_coeff = dict()
        # init parameter dictionary
        self.dict_param = dict()
        self._init_dict()

        self.scale_cons = scale_cons
        self.scale_prod = scale_prod

        reduce_qty_data=int(reduce_qty_data) if reduce_qty_data is not None else int(1)
        length_history=int(length_history) if length_history is not None else int(12)
        start_history=int(start_history) if start_history is not None else int(0)


        # Defining the type of environment
        self._dist_equinox = dist_equinox
        self._pred = pred
        self._reduce_qty_data=reduce_qty_data   # Factor by which to artificially reduce the data available (for training+validation)
                                                # Choices are 1,2,4,8,16
                                                
        self._length_history=max(1,length_history)     # Length for the truncature of the history to build the pseudo-state

        self._start_history=start_history       # Choice between data that is replicated (choices are in [0,...,self._reduce_qty_data[ )

        self._max_ener_buy = max_ener_buy
        self._max_ener_sell = max_ener_sell
        
        if (self._dist_equinox and self._pred):
            self._last_ponctual_observation = [0.] \
                                              + [0. for _ in range(self._length_history * 2 + 3)]
        elif (self._dist_equinox and not self._pred):
            self._last_ponctual_observation = [0.] + [0. for _ in range(self._length_history * 2 + 1)]
        elif (not self._dist_equinox and not self._pred):
            self._last_ponctual_observation = [0.] + [0. for _ in range(self._length_history * 2)]
        self._input_dimensions = [(len(self._last_ponctual_observation),)]
        self._init_gym()

        self._rng = rng

        # Get consumption profile in [0,1]
        absolute_dir = os.path.dirname(os.path.abspath(__file__))
        if consumption is None:
            consumption=np.load(absolute_dir + "/data/example_nondeterminist_cons_train.npy")[0:1*365*24]
        self.consumption_norm=consumption
        #self.consumption_valid_norm=np.load(absolute_dir + "/data/example_nondeterminist_cons_train.npy")[365*24:2*365*24]
        #self.consumption_test_norm=np.load(absolute_dir + "/data/example_nondeterminist_cons_test.npy")[0:1*365*24]
        # Scale consumption profile in [0,2.1kW] --> average max per day = 1.7kW, average per day is 18.3kWh
        self.consumption=self.consumption_norm*scale_cons
        #self.consumption_valid=self.consumption_valid_norm*2.1
        #self.consumption_test=self.consumption_test_norm*2.1

        self.min_consumption=min(self.consumption)
        self.max_consumption=max(self.consumption)
        if verbose:
            print("Sample of the consumption profile (kW): {}".format(self.consumption[0:24]))
            print("Min of the consumption profile (kW): {}".format(self.min_consumption))
            print("Max of the consumption profile (kW): {}".format(self.max_consumption))
            print("Average consumption per day train (kWh): {}".format(np.sum(self.consumption)/self.consumption.shape[0]*24))

        if production is None:
            production = np.load(absolute_dir + "/data/BelgiumPV_prod_train.npy")[0:1*365*24]
        # Get production profile in W/Wp in [0,1]
        self.production_norm = production
        #self.production_valid_norm=np.load(absolute_dir + "/data/BelgiumPV_prod_train.npy")[365*24:2*365*24] #determinist best is 110, "nondeterminist" is 124.9
        #self.production_test_norm=np.load(absolute_dir + "/data/BelgiumPV_prod_test.npy")[0:1*365*24] #determinist best is 76, "nondeterminist" is 75.2
        # Scale production profile : 12KWp (60m^2) et en kWh
        self.production=self.production_norm*scale_prod

        if verbose:
            print ("self.production_train brefore")
            print (self.production)
        
        ###
        ### Artificially reducing the variety of the training and validation time series
        ###
        # We create the largest 4 consecutive blocs of days  (seasons)
        # so that the number of days is divisible by self._reduce_qty_data
        # Then we within each season, we reduce the qty of days that are different
        if(self._reduce_qty_data==2):
            nd_one_seas=90*24
        elif(self._reduce_qty_data==4 or self._reduce_qty_data==8):
            nd_one_seas=88*24
        elif(self._reduce_qty_data==16):
            nd_one_seas=80*24
        
        if(self._reduce_qty_data!=1):
            for season in range(4):
                self.production[season*nd_one_seas:(season+1)*nd_one_seas]=np.tile(
                    self.production[int((season+(self._start_history+0.)/self._reduce_qty_data)*nd_one_seas):
                                    int((season+(self._start_history+1.)/self._reduce_qty_data)*nd_one_seas)],
                    self._reduce_qty_data)
                self.production_norm[season*nd_one_seas:(season+1)*nd_one_seas]=np.tile(
                    self.production_norm[int((season+(self._start_history+0.)/self._reduce_qty_data)*nd_one_seas):
                                         int((season+(self._start_history+1.)/self._reduce_qty_data)*nd_one_seas)],
                    self._reduce_qty_data)
        if verbose:
            print ("self.production_train after")
            print (self.production)

        if verbose:
            self.min_production=min(self.production)
            self.max_production=max(self.production)
            print("Sample of the production profile (kW): {}".format(self.production[0:24]))
            print("Min of the production profile (kW): {}".format(self.min_production))
            print("Max of the production profile (kW): {}".format(self.max_production))
            print("Average production per day train (kWh): {}".format(np.sum(self.production)/self.production.shape[0]*24))

        self.battery_size=15.
        self.battery_eta=0.9
        
        self.hydrogen_max_power=1.1
        self.hydrogen_eta=.65

    def _init_dict(self):
        self.dict_param["flow_H2"] = 0.       # Value between 0 and 1
        self.dict_param["flow_lithium"] = 0.  # Value between 0 and 1
        self.dict_param["lack_energy"] = 0.   # Value between 0 and 1
        self.dict_param["waste_energy"] = 0.  # Value between 0 and 1
        self.dict_param["soc"] = 0.           # Value between 0 and 1
        self.dict_param["buy_energy"] = 0.    # Value between 0 and 1
        self.dict_param["sell_energy"] = 0.   # Value between 0 and 1

    def reset(self):
        """
        Returns:
           current observation (list of k elements)
        """
        self._init_dict()
        self.save_state.append(dict())
        ### Test 6
        if (self._dist_equinox and self._pred):
            self._last_ponctual_observation = [1.]+ [0. for _ in range(self._length_history * 2 + 3)]
        elif (self._dist_equinox and not self._pred):
            self._last_ponctual_observation = [1.]+ [0. for _ in range(self._length_history * 2 + 1)]
        elif (not self._dist_equinox and not self._pred):
            self._last_ponctual_observation = [1.]+ [0. for _ in range(self._length_history * 2)]

        self.counter = 1        
        self.hydrogen_storage=0.


        if (self._dist_equinox and self._pred):
            return np.array([0.] + [0. for _ in range(self._length_history * 2 + 3)])
        elif (self._dist_equinox and not self._pred):
            return np.array([0.] + [0. for _ in range(self._length_history * 2 + 1)])
        else: #elif (not self._dist_equinox, not self._pred):
            return np.array([0.] + [0. for _ in range(self._length_history * 2)])

    def step(self, action):
        """
        Perform one time step on the environment
        """
        self._init_dict()
        true_demand=self.consumption[self.counter-1]-self.production[self.counter-1]

        if (action==0):
            ## Energy is taken out of the hydrogen reserve
            energy = min(self.hydrogen_max_power, self.hydrogen_storage)
            true_energy_avail_from_hydrogen=-energy*self.hydrogen_eta
            diff_hydrogen=-energy
        if (action==1):
            ## No energy is taken out of/into the hydrogen reserve
            true_energy_avail_from_hydrogen=0
            diff_hydrogen=0
        if (action==2):
            ## Energy is taken into the hydrogen reserve
            true_energy_avail_from_hydrogen=self.hydrogen_max_power/self.hydrogen_eta
            diff_hydrogen=self.hydrogen_max_power

        self.dict_param["flow_H2"] = diff_hydrogen / self.hydrogen_max_power
        self.hydrogen_storage+=diff_hydrogen

        Energy_needed_from_battery=true_demand+true_energy_avail_from_hydrogen
        
        if (Energy_needed_from_battery>0):
            # Lack of energy
            if (self._last_ponctual_observation[0]*self.battery_size>Energy_needed_from_battery):
                # If enough energy in the battery, use it
                self.dict_param["lack_energy"] = 0
                self.dict_param["flow_lithium"] = -Energy_needed_from_battery/self.battery_eta/self.battery_size
                self._last_ponctual_observation[0] = self._last_ponctual_observation[0] + \
                                                     self.dict_param["flow_lithium"]

            else:
                # Otherwise: use what is left and then penalty
                self.dict_param["lack_energy"] = (Energy_needed_from_battery -
                                                  self._last_ponctual_observation[0] * self.battery_size)
                self.dict_param["flow_lithium"] = - self._last_ponctual_observation[0]
                self._last_ponctual_observation[0] = 0
            if self._max_ener_buy is not None:
                self.dict_param["buy_energy"] = min(self._max_ener_buy, self.dict_param["lack_energy"])
            else:
                self.dict_param["buy_energy"] = self.dict_param["lack_energy"]
            self.dict_param["lack_energy"] -= self.dict_param["buy_energy"]

        elif (Energy_needed_from_battery < 0):
            # Surplus of energy --> load the battery
            self.dict_param["waste_energy"] = max(0, (self._last_ponctual_observation[0] * self.battery_size
                                                  - Energy_needed_from_battery * self.battery_eta) - self.battery_size)
            tmp_bat = self._last_ponctual_observation[0]
            self._last_ponctual_observation[0] = min(1.,
                    self._last_ponctual_observation[0]-Energy_needed_from_battery/self.battery_size*self.battery_eta)
            self.dict_param["flow_lithium"] = (self._last_ponctual_observation[0] - tmp_bat)
        if self._max_ener_sell is not None:
            self.dict_param["sell_energy"] = min(self._max_ener_sell, self.dict_param["waste_energy"])
        else:
            self.dict_param["sell_energy"] = self.dict_param["waste_energy"]
        self.dict_param["waste_energy"] -= self.dict_param["sell_energy"]

        self._last_ponctual_observation[1:self._length_history] = self._last_ponctual_observation[2:self._length_history+1]
        self._last_ponctual_observation[self._length_history] = self.consumption_norm[self.counter]
        self._last_ponctual_observation[self._length_history + 1:2 * self._length_history] = \
            self._last_ponctual_observation[self._length_history + 2: 2 * self._length_history+1]
        self._last_ponctual_observation[2 * self._length_history] = self.production_norm[self.counter]


        self.dict_param["soc"] = self._last_ponctual_observation[0]

        i=2 * self._length_history
        #i=1
        if(self._dist_equinox):
            i=i+1
            self._last_ponctual_observation[i]=abs(self.counter/24-(365./2))/(365./2) # 171 days between 1 jan and 21 Jun
        if (self._pred):
            i=i+1
            self._last_ponctual_observation[i] = sum(self.production_norm[self.counter:self.counter+24])/24.
            self._last_ponctual_observation[i+1] = sum(self.production_norm[self.counter:self.counter+48])/48.
                                
        self.counter+=1

        #normalized value
        self.dict_param["flow_H2"] = (1. + self.dict_param["flow_H2"]) / 2.
        self.dict_param["flow_lithium"] = (1. + self.dict_param["flow_lithium"]) / 2.
        self.dict_param["lack_energy"] /= (self.scale_cons + self.hydrogen_max_power)
        self.dict_param["lack_energy"] /= (self.scale_cons + self.hydrogen_max_power)
        self.dict_param["buy_energy"] /= (self.scale_cons + self.hydrogen_max_power)
        self.dict_param["sell_energy"] /= (self.scale_prod + self.hydrogen_max_power)
        self.dict_param["waste_energy"] /= (self.scale_prod + self.hydrogen_max_power) 

        dict_reward = self.my_reward()
        if self._pred:
            done = self.counter + 24 >= len(self.production_norm)
        else:
            done = self.counter >= len(self.production_norm)
        info = {}
        reward = np.sum(list(dict_reward.values()))



        self._save({"action": action,
                    "rewards": reward,"consumption": self.consumption[self.counter-1],
                    "production": self.production[self.counter-1], "battery_h2": self.hydrogen_storage})
        self._save(self.dict_param)
        self._save(dict_reward)

        return np.array(copy.deepcopy(self._last_ponctual_observation)), reward, done, info

    def _save(self, d_state):
        for key, val in d_state.items():
            if key not in self.save_state[-1]:
                self.save_state[-1][key] = []
            self.save_state[-1][key].append(val)


    def add_reward(self, key, fn, coeff=1.):
        self.dict_coeff[key] = coeff
        self.dict_reward[key] = fn

    def my_reward(self):
        """
        return the test result and the list of test values not used for training
        """
        """res = dict()
        for key, fn in self.dict_reward.items():
            val_fn = fn(self.dict_param)
            res[key] = val_fn * self.dict_coeff[key]
        return res"""
        return {key: fn(self.dict_param) * self.dict_coeff[key] for key, fn in self.dict_reward.items()}

    def get_data(self):
        return self.save_state

    def _init_gym(self):
        self.action_space = Discrete(3)
        if (self._dist_equinox and self._pred):
            self.observation_space = Box(low=np.zeros((1 + self._length_history * 2 + 3)),
                                         high=np.ones((1 + self._length_history * 2 + 3)))
        elif (self._dist_equinox and not self._pred):
            self.observation_space = Box(low=np.zeros((1 + self._length_history * 2 + 1)),
                                         high=np.ones((1 + self._length_history * 2 + 1)))

        elif (not self._dist_equinox and not self._pred):
            self.observation_space = Box(low=np.zeros((1 + self._length_history * 2)),
                                     high=np.ones((1 + self._length_history * 2)))
