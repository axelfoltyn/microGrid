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

"""
import random

import numpy as np
import copy
import os

import gym
from gym.spaces import Discrete, Box, Tuple
from deer.base_classes import Environment
from microGrid.env.plot_MG_operation import plot_op


class MicroGrid2(Environment, gym.Env):
    def __init__(self, rng, reduce_qty_data=None, length_history=None, start_history=None, production_norm=None,
                 consumption_norm=None, scale_prod=12000. / 1000 / 2, scale_cons=2.1, random_start=True):
        """ Initialize environment

        Arguments:
            rng - the numpy random number generator
        """
        # random start condition to improve learning
        self._random_start = random_start
        absolute_dir = os.path.dirname(os.path.abspath(__file__))
        if consumption_norm is None:
            consumption_norm = np.load(absolute_dir + "/data/example_nondeterminist_cons_train.npy")[
                                          0:1 * 365 * 24]
        if production_norm is None:
            production_norm = np.load(absolute_dir+"/data/BelgiumPV_prod_train.npy")[0:1 * 365 * 24]
        self.production_norm = production_norm
        self.consumption_norm = consumption_norm
        self.consumption = self.consumption_norm * scale_cons
        self.production = self.production_norm * scale_prod

        reduce_qty_data = int(reduce_qty_data) if reduce_qty_data is not None else int(1)
        length_history = int(length_history) if length_history is not None else int(12)
        #length_history = 1
        start_history = int(start_history) if start_history is not None else int(0)
        print("reduce_qty_data, length_history, start_history")
        print(reduce_qty_data, length_history, start_history)

        # dict_reward stores the reward function
        # dict_coeff stores the reward coefficient
        # (used in add_reward and my_reward)
        self.dict_reward = dict()
        self.dict_coeff = dict()
        # init parameter dictionary
        self._init_dict()

        # Factor by which to artificially reduce the data available (for training+validation)
        self._reduce_qty_data = reduce_qty_data
        # Choices are 1,2,4,8,16

        # Length for the truncature of the history to build the pseudo-state
        self._length_history = length_history

        # Choice between data that is replicated (choices are in [0,...,self._reduce_qty_data[ )
        self._start_history = start_history

        inc_sizing = 1.

        # Defining the type of environment
        self._dist_equinox = 0
        self._pred = 0
        self._init_obs()

        # use for openai env
        self.action_space = Discrete(3)

        self._rng = rng

        self.min_consumption = min(self.consumption)
        self.max_consumption = max(self.consumption)
        print("Sample of the consumption profile (kW): {}".format(self.consumption[0:24]))
        print("Min of the consumption profile (kW): {}".format(self.min_consumption))
        print("Max of the consumption profile (kW): {}".format(self.max_consumption))
        print("Average consumption per day (kWh): {}".format(
            np.sum(self.consumption) / self.consumption.shape[0] * 24))

        print("self.production brefore")
        print(self.production)

        ###
        ### Artificially reducing the variety of the training and validation time series
        ###
        # We create the largest 4 consecutive blocs of days  (seasons)
        # so that the number of days is divisible by self._reduce_qty_data
        # Then we within each season, we reduce the qty of days that are different
        if (self._reduce_qty_data == 2):
            nd_one_seas = 90 * 24
        elif (self._reduce_qty_data == 4 or self._reduce_qty_data == 8):
            nd_one_seas = 88 * 24
        elif (self._reduce_qty_data == 16):
            nd_one_seas = 80 * 24
        elif(self._reduce_qty_data!= 1):
            print("reduce_qty_data: Factor by which to artificially reduce the data available (for training+validation)")
            print("Choices are 1,2,4,8,16")
            raise

        if (self._reduce_qty_data != 1):
            for season in range(4):
                self.production[season * nd_one_seas:(season + 1) * nd_one_seas] = np.tile(self.production[
                                     int((season + (
                                                 self._start_history + 0.) / self._reduce_qty_data) * nd_one_seas):int(
                                         (season + (
                                                     self._start_history + 1.) / self._reduce_qty_data) * nd_one_seas)],
                                     self._reduce_qty_data)

                self.production_norm[season * nd_one_seas:(season + 1) * nd_one_seas] = np.tile(
                    self.production_norm[
                    int((season + (self._start_history + 0.) / self._reduce_qty_data) * nd_one_seas):int(
                        (season + (self._start_history + 1.) / self._reduce_qty_data) * nd_one_seas)],
                    self._reduce_qty_data)

        print("self.production after")
        print(self.production)

        self.min_production = min(self.production)
        self.max_production = max(self.production)
        print("Sample of the production profile (kW): {}".format(self.production[0:24]))
        print("Min of the production profile (kW): {}".format(self.min_production))
        print("Max of the production profile (kW): {}".format(self.max_production))
        print("Average production per day train (kWh): {}".format(
            np.sum(self.production) / self.production.shape[0] * 24))

        self.battery_size = 15. * inc_sizing
        self.battery_max_power = 1.1 * inc_sizing
        self.battery_eta = 0.9

        self.hydrogen_max_power = 1.1 * inc_sizing
        self.hydrogen_eta = .65

    ##############################################################
    ##############################################################
    ###########              gym function             ############
    ##############################################################
    ##############################################################

    def step(self, action):
        true_demand = self.consumption[self.counter - 1] - self.production[self.counter - 1]

        # action 0 discharges H2 reserve  1 do nothing and 2 charge H2 reserve
        true_energy_avail_from_hydrogen = (action - 1) * self.hydrogen_max_power * self.hydrogen_eta
        diff_hydrogen = (action - 1) * self.hydrogen_max_power

        self.dict_param["flow_H2"] = diff_hydrogen
        self.hydrogen_storage += diff_hydrogen

        Energy_needed_from_battery = true_demand + true_energy_avail_from_hydrogen

        if (Energy_needed_from_battery > 0):
            # Lack of energy
            energy_take = self.battery_max_power
            if (energy_take > Energy_needed_from_battery):
                # If enough energy in the battery, use it
                self.dict_param["lack_energy"] = 0
                self._last_ponctual_observation[0] -= Energy_needed_from_battery / self.battery_size / self.battery_eta
            else:
                # Otherwise: use max power and then penalty
                self.dict_param["lack_energy"] = Energy_needed_from_battery - energy_take
                self._last_ponctual_observation[0] -= energy_take / self.battery_size / self.battery_eta
        elif (Energy_needed_from_battery < 0):
            # Surplus of energy --> load the battery
            self.dict_param["waste_energy"] = max(0, (self._last_ponctual_observation[0] * self.battery_size -
                                                  Energy_needed_from_battery * self.battery_eta) - self.battery_size)
            self._last_ponctual_observation[0] = min(1., self._last_ponctual_observation[
                0] - Energy_needed_from_battery / self.battery_size * self.battery_eta)

        # print "new self._last_ponctual_observation[0]"
        # print self._last_ponctual_observation[0]

        ### Test
        # self._last_ponctual_observation[0] : State of the battery (0=empty, 1=full)
        # self._last_ponctual_observation[1][0] : Normalized consumption at current time step (-> not available at decision time)
        # self._last_ponctual_observation[1][1] : Normalized production at current time step (-> not available at decision time)
        # self._last_ponctual_observation[2] : distance to equinox
        # self._last_ponctual_observation[2 or 3][0] : Prevision (accurate) for the current time step and the next 24hours
        # self._last_ponctual_observation[3 or 4][1] : Prevision (accurate) for the current time step and the next 48hours
        ###
        self._last_ponctual_observation[1][0] = self.consumption_norm[self.counter]
        self._last_ponctual_observation[1][1] = self.production_norm[self.counter]
        i = 1
        if self._dist_equinox == 1:
            i = i + 1
            self._last_ponctual_observation[i] = abs(self.counter / 24 - (365. / 2)) / (
                    365. / 2)  # 171 days between 1jan and 21 Jun
        if self._pred == 1:
            i = i + 1
            self._last_ponctual_observation[i][0] = sum(
                self.production_norm[self.counter:self.counter + 24]) / 24.  # *self.rng.uniform(0.75,1.25)
            self._last_ponctual_observation[i][1] = sum(
                self.production_norm[self.counter:self.counter + 48]) / 48.  # *self.rng.uniform(0.75,1.25)


        self.counter += 1
        dict_reward = self.my_reward()
        done = self.counter + 24 >= len(self.production_norm) if self._pred == 1 else self.counter >= len(self.production_norm)
        info = {}
        print("===================>",self._last_ponctual_observation)
        return self._last_ponctual_observation, np.sum(list(dict_reward.values())), done, info

    def render(self, mode='human', close=False):
        pass

    ##############################################################
    ##############################################################
    ###########             deerv2 function             ############
    ##############################################################
    ##############################################################

    def observationType(self, subject):
        """Gets the most inner type (np.uint8, np.float32, ...) of [subject].

        Parameters
        -----------
        subject : int
            The subject
        """

        return np.float64

    def _init_dict(self):
        self.dict_param = dict()
        self.dict_param["flow_H2"] = 0
        self.dict_param["lack_energy"] = 0
        self.dict_param["waste_energy"] = 0

    def reset(self):
        """
        Returns:
           current observation (list of k elements)
        """
        ### Test 6
        if (self._dist_equinox == 1 and self._pred == 1):
            self._last_ponctual_observation = [1., [0., 0.], 0., [0., 0.]]
        elif (self._dist_equinox == 1 and self._pred == 0):
            self._last_ponctual_observation = [1., [0., 0.], 0.]
        elif (self._dist_equinox == 0 and self._pred == 0):
            self._last_ponctual_observation = [1., [0., 0.]]

        if self._random_start:
            self.counter = random.randint(1, len(self.production_norm)-1)
        else:
            self.counter = 1
        self.hydrogen_storage = 0.

        if (self._dist_equinox == 1 and self._pred == 1):
            return [
                0.,
                [[0., 0.] for _ in range(self._length_history)],
                0.,
                [0., 0.]
            ]
        elif (self._dist_equinox == 1 and self._pred == 0):
            return [
                0.,
                [[0., 0.] for _ in range(self._length_history)],
                0.
            ]
        else:  # elif (self._dist_equinox==0, self._pred==0):
            return [
                0.,
                [[0., 0.] for _ in range(self._length_history)],
            ]

    def act(self, action):
        """
        Perform one time step on the environment
        """
        _, reward, _, _ = self.step(action)
        return reward

    def inputDimensions(self):
        return self._input_dimensions

    def nActions(self):
        return self.action_space.n

    def observe(self):
        return copy.deepcopy(self._last_ponctual_observation)

    #TODO not use that
    """def summarizePerformance(self, test_data_set, *args, **kwargs):
        print("summary perf")
        print("self.hydrogen_storage: {}".format(self.hydrogen_storage))
        observations = test_data_set.observations()
        actions_test = test_data_set.actions()
        rewards = test_data_set.rewards()
        actions = []
        for action in actions_test:
            if (action == 0):
                actions.append(-self.hydrogen_max_power)
            elif (action == 1):
                actions.append(0)
            elif (action == 2):
                actions.append(self.hydrogen_max_power)

        battery_level = np.array(observations[0]) * self.battery_size
        consumption = np.array(observations[1][:, 0]) * (
                    self.max_consumption - self.min_consumption) + self.min_consumption
        production = np.array(observations[1][:, 1]) * (self.max_production - self.min_production) + self.min_production

        i = 0
        plot_op(actions[0 + i:100 + i], consumption[0 + i:100 + i], production[0 + i:100 + i], rewards[0 + i:100 + i],
                battery_level[0 + i:100 + i], "plot_winter_.png")
        #
        i = 180 * 24
        plot_op(actions[0 + i:100 + i], consumption[0 + i:100 + i], production[0 + i:100 + i], rewards[0 + i:100 + i],
                battery_level[0 + i:100 + i], "plot_summer_.png")
        #
        i = 360 * 24
        plot_op(actions[0 + i:100 + i], consumption[0 + i:100 + i], production[0 + i:100 + i], rewards[0 + i:100 + i],
                battery_level[0 + i:100 + i], "plot_winter2_.png")"""

        ##############################################################
        ##############################################################
        ###########              my function              ############
        ##############################################################
        ##############################################################

    def add_reward(self, key, fn, coeff):
        self.dict_coeff[key] = coeff
        self.dict_reward[key] = fn

    def my_reward(self):
        """
        return the test result and the list of test values not used for training
        """
        res = dict()
        for key, fn in self.dict_reward.items():
            val_fn = fn(self.dict_param)
            res[key] = val_fn * self.dict_coeff[key]
        return res

    def _init_obs(self):
        # part openai.gym
        if (self._dist_equinox == 1 and self._pred == 1):
            self.observation_space = Tuple([
                Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float64),
                Box(low=np.zeros((self._length_history, 2)), high=np.ones((self._length_history, 2)),
                    dtype=np.float64),
                Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float64),
                Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float64),
                Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float64)
                ])
        elif (self._dist_equinox == 1 and self._pred == 0):
            self.observation_space = Tuple([
                Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float64),
                Box(low=np.zeros((self._length_history, 2)), high=np.ones((self._length_history, 2)),
                    dtype=np.float64),
                Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float64)
            ])
        elif (self._dist_equinox == 0 and self._pred == 0):
            self.observation_space = Tuple([
                Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float64),
                Box(low=np.zeros((self._length_history, 2)), high=np.ones((self._length_history, 2)),
                    dtype=np.float64)
            ])
        #part deerv2
        if (self._dist_equinox == 1 and self._pred == 1):
            self._last_ponctual_observation = [0., [0., 0.], 0., [0., 0.]]
            self._input_dimensions = [(1,), (self._length_history, 2), (1,), (1, 2)]
        elif (self._dist_equinox == 1 and self._pred == 0):
            self._last_ponctual_observation = [0., [0., 0.], 0.]
            self._input_dimensions = [(1,), (self._length_history, 2), (1,)]
        elif (self._dist_equinox == 0 and self._pred == 0):
            self._last_ponctual_observation = [0., [0., 0.]]
            self._input_dimensions = [(1,), (self._length_history, 2)]

    def getinput(self):
        l_obs = list(self.observation_space.sample())
        res = []
        for i in range(len(l_obs)):
            if hasattr(l_obs[i], '__len__'):
                res.append((1, len(l_obs[i])))
            else:
                res.append((1,))
        return res

def main():
    rng = np.random.RandomState(123456)
    myenv = MicroGrid2(rng)

    print(myenv.observe())


if __name__ == "__main__":
    main()
