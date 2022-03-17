from gym.spaces import Discrete, Box, Dict, Tuple
import gym
import numpy as np
import os

"""
modified Vincent Francois-Lavet code
"""
class MicroGridEnv (gym.Env):
    """
    the environment representing a micro electrical network.
    There are three actions available.
    0 : charge the hydrogen storage
    1 : do nothing
    2 : unload the hydrogen storage

    For the reward functions, the following format must be respected:
    fn(dico_param)
    the dico_param will store the values of the possible parameters
    used in this function it will return a float
    """
    def __init__(self, reduce_qty_data=None, length_history=None, start_history=None):

        self.VALIDATION_MODE = 0
        self.TEST_MODE = 1
        self._mode = -1

        self.action_space = Discrete(3)
        # Length for the truncature of the history to build the pseudo-state
        length_history = int(length_history) if length_history is not None else int(12)
        self._length_history = length_history
        self._input_dimensions = None
        self._dist_equinox = 1
        self._pred = 1
        self._init_obs()

        # dict_reward stores the reward function
        # dict_coeff stores the reward coefficient
        # (used in add_reward and my_reward)
        self.dict_reward = dict()
        self.dict_coeff_train = dict()
        self.dict_coeff_test = dict()
        #init parameter dictionary
        self._init_dict()

        inc_sizing = 1.
        self._load_data(inc_sizing)

        self.battery_max_power = 1.1 * inc_sizing
        self.battery_size = 15. * inc_sizing
        self.battery_eta = 0.9

        self.hydrogen_max_power = 1.1 * inc_sizing
        self.hydrogen_eta = .65

    def observationType(self, subject):
        """Gets the most inner type (np.uint8, np.float32, ...) of [subject].

        Parameters
        -----------
        subject : int
            The subject
        """

        return np.float64

    def nActions(self):
        return self.action_space.n

    def _init_dict(self):
        self.dict_param = dict()
        self.dict_param["flow_H2"] = 0
        self.dict_param["lack_energy"] = 0
        self.dict_param["waste_energy"] = 0

    def _load_data(self, inc_sizing):

        # Get consumption profile in [0,1]
        absolute_dir = os.path.dirname(os.path.abspath(__file__))
        self.consumption_train_norm = np.load(absolute_dir+"/data/example_nondeterminist_cons_train.npy")[0:1 * 365 * 24]
        self.consumption_valid_norm = np.load(absolute_dir+"/data/example_nondeterminist_cons_train.npy")[365 * 24:2 * 365 * 24]
        self.consumption_test_norm = np.load(absolute_dir+"/data/example_nondeterminist_cons_test.npy")[0:1 * 365 * 24]
        # Scale consumption profile in [0,2.1kW] --> average max per day = 1.7kW, average per day is 18.3kWh
        self.consumption_train = self.consumption_train_norm * 2.1
        self.consumption_valid = self.consumption_valid_norm * 2.1
        self.consumption_test = self.consumption_test_norm * 2.1

        # Get production profile in W/Wp in [0,1]
        self.production_train_norm = np.load(absolute_dir+"/data/BelgiumPV_prod_train.npy")[0:1 * 365 * 24] / 2
        self.production_valid_norm = np.load(absolute_dir+"/data/BelgiumPV_prod_train.npy")[
                                     365 * 24:2 * 365 * 24] / 2  # determinist best is 110, "nondeterminist" is 124.9
        self.production_test_norm = np.load(absolute_dir+"/data/BelgiumPV_prod_test.npy")[
                                    0:1 * 365 * 24] / 2  # determinist best is 76, "nondeterminist" is 75.2
        # Scale production profile : 12KWp (60m^2) et en kWh
        self.production_train = self.production_train_norm * 12000. / 1000. * inc_sizing
        self.production_valid = self.production_valid_norm * 12000. / 1000. * inc_sizing
        self.production_test = self.production_test_norm * 12000 / 1000 * inc_sizing

    def inputDimensions(self):
        if self._input_dimensions is None:
            self._input_dimensions = self.getinput()

        return self._input_dimensions

    def getinput(self):
        l_obs = list(self.observation_space.sample())
        res = []
        for i in range(len(l_obs)):
            if hasattr(l_obs[i], '__len__'):
                res.append((1, len(l_obs[i])))
            else:
                res.append((1,))
        return res

    def _init_obs(self):
        if (self._dist_equinox == 1 and self._pred == 1):
            self.observation_space = Tuple([
                Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float64),
                Box(low=np.zeros((self._length_history)), high=np.ones((self._length_history)),
                    dtype=np.float64),
                Box(low=np.zeros((self._length_history)), high=np.ones((self._length_history)),
                    dtype=np.float64),
                Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float64),
                Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float64),
                Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float64)
                ])
        elif (self._dist_equinox == 1 and self._pred == 0):
            self.observation_space = Tuple([
                Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float64),
                Box(low=np.zeros((self._length_history)), high=np.ones((self._length_history)),
                                   dtype=np.float64),
                Box(low=np.zeros((self._length_history)), high=np.ones((self._length_history)),
                                  dtype=np.float64),
                Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float64)
            ])
        elif (self._dist_equinox == 0 and self._pred == 0):
            self.observation_space = Tuple([
                Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float64),
                Box(low=np.zeros((self._length_history)), high=np.ones((self._length_history)),
                                   dtype=np.float64),
                Box(low=np.zeros((self._length_history)), high=np.ones((self._length_history)),
                                  dtype=np.float64)
            ])

    def step(self, action):
        reward = 0  # self.ale.act(action)  #FIXME
        terminal = 0

        # TODO how choice the mode
        true_demand = self.consumption[self.counter - 1] - self.production[self.counter - 1]

        # action 0 discharges H2 reserve  1 do nothing and 2 charge H2 reserve
        true_energy_avail_from_hydrogen = (action - 1) * self.hydrogen_max_power * self.hydrogen_eta
        diff_hydrogen = (action - 1) * self.hydrogen_max_power


        self.dict_param["flow_H2"] = diff_hydrogen
        self.hydrogen_storage += diff_hydrogen

        Energy_needed_from_battery = true_demand + true_energy_avail_from_hydrogen

        if (Energy_needed_from_battery > 0):
            # Lack of energy
            if (self.observation_space[0] * self.battery_size > Energy_needed_from_battery):
                # If enough energy in the battery, use it
                self.dict_param["lack_energy"] = 0
                self.observation_space[0] = self.observation_space[0] \
                                                     - Energy_needed_from_battery / self.battery_size / self.battery_eta
            else:
                # Otherwise: use what is left and then penalty
                self.dict_param["lack_energy"] = (Energy_needed_from_battery - self._last_ponctual_observation[
                    0] * self.battery_size)
                self.observation_space[0] = 0
        elif (Energy_needed_from_battery < 0):
            # Surplus of energy --> load the battery
            self.dict_param["waste_energy"] = max(0, (self.observation_space[0] * self.battery_size
                                             - Energy_needed_from_battery * self.battery_eta) - self.battery_size)
            self.observation_space[0] = min(1., self.observation_space[0]
                                            - Energy_needed_from_battery / self.battery_size * self.battery_eta)

        # print "new self._last_ponctual_observation[0]"
        # print self._last_ponctual_observation[0]

        ### Test
        # self._last_ponctual_observation[0] : State of the battery (0=empty, 1=full)
        # self._last_ponctual_observation[1] : Normalized consumption at current time step (-> not available at decision time)
        # self._last_ponctual_observation[1][1] : Normalized production at current time step (-> not available at decision time)
        # self._last_ponctual_observation[2][0] : Prevision (accurate) for the current time step and the next 24hours
        # self._last_ponctual_observation[2][1] : Prevision (accurate) for the current time step and the next 48hours
        ###
        #self._last_ponctual_observation[1][0] = self.consumption_norm[self.counter]
        self.observation_space[1] = self.consumption_norm[self.counter]
        #self._last_ponctual_observation[1][1] = self.production_norm[self.counter]
        self.observation_space[2] = self.production_norm[self.counter]
        #i = 1
        if (self._dist_equinox == 1):
            self.observation_space[3] = abs(self.counter / 24 - (365. / 2)) / (
                        365. / 2)  # 171 days between 1jan and 21 Jun
        if (self._pred == 1):
            self.observation_space[4] = sum(
                self.production_norm[self.counter:self.counter + 24]) / 24.  # *self.rng.uniform(0.75,1.25)
            self.observation_space[5] = sum(
                self.production_norm[self.counter:self.counter + 48]) / 48.  # *self.rng.uniform(0.75,1.25)
        self.counter += 1

        done = self.counter > len(self.production_norm)
        info = self.dict_param

        return self.observation_space, self.my_reward(), done, info

    def render(self, mode="human"):
        self._print()
        pass

    def _print(self, agent):
        if self._periodicity <= 1 or self._count % self._periodicity == 0:
            print("{} {}:".format(self._string, self._count + 1))
            print("Learning rate: {}".format(agent._learning_algo.learningRate()))
            print("Discount factor: {}".format(agent._learning_algo.discountFactor()))
            print("Epsilon: {}".format(agent._train_policy.epsilon()))

    def set_mode(self, mode):
        self._mode = mode

    def reset(self):
        ### Test 6
        # [%charge bat lithium, [prod, conso], equinox, [prediction_prod, prediction_cons]]
        if (self._dist_equinox == 1 and self._pred == 1):
            self._last_ponctual_observation = [1., [0., 0.], 0., [0., 0.]]
        elif (self._dist_equinox == 1 and self._pred == 0):
            self._last_ponctual_observation = [1., [0., 0.], 0.]
        elif (self._dist_equinox == 0 and self._pred == 0):
            self._last_ponctual_observation = [1., [0., 0.]]

        self.counter = 1
        self.hydrogen_storage = 0.

        if self._mode == "train":
            self.production_norm=self.production_train_norm
            self.production=self.production_train
            self.consumption_norm=self.consumption_train_norm
            self.consumption=self.consumption_train
        elif self._mode == self.VALIDATION_MODE:
            self.production_norm=self.production_valid_norm
            self.production=self.production_valid
            self.consumption_norm=self.consumption_valid_norm
            self.consumption=self.consumption_valid
        else:
            self.production_norm=self.production_test_norm
            self.production=self.production_test
            self.consumption_norm=self.consumption_test_norm
            self.consumption=self.consumption_test

        if (self._dist_equinox==1 and self._pred==1):
            return [
                        0.,
                        [[0., 0.] for i in range(self._length_history)],
                        0.,
                        [0., 0.]
                    ]
        if (self._dist_equinox==1 and self._pred==0):
            return [
                        0.,
                        [[0., 0.] for i in range(self._length_history)],
                        0.
                    ]
        return [
                        0.,
                        [[0., 0.] for i in range(self._length_history)],
                    ]

    def add_reward(self, key, fn, coeff_train, coeff_test=None):
        if coeff_test is None:
            coeff_test = coeff_train
        self.dict_coeff_train[key] = coeff_train
        self.dict_coeff_test[key] = coeff_test
        self.dict_reward[key] = fn

    def my_reward(self):
        """
        return the test result and the list of test values not used for training
        """
        res_train = dict()
        res_test = dict()
        for key, fn in self.dict_reward.items():
            val_fn = fn(self.dict_param)
            res_train[key] = val_fn * self.dict_coeff_train[key]
            res_test[key] = val_fn * self.dict_coeff_test[key]
        return res_train, res_test


if __name__ == '__main__':
    env = MicroGridEnv()
    obs = env.observation_space.sample()
    print(obs)
    print(obs[0])
    price_h2 = 0.1  # 0.1euro/kWh of hydrogen
    price_elec_buy = 2.0 # 2euro/kWh
    env.add_reward("flow_H2", lambda x: x["flow_H2"] * price_h2, 1., 1.)
    env.add_reward("buy_energy", lambda x: -x["lack_energy"] * price_elec_buy, 1., 1.)
    env.add_reward("flow_H2_bias", lambda x: x["flow_H2"] * price_h2 + 1, 1., 2.)

    dict_train, dict_test = env.my_reward()
    print(dict_train, sum(dict_train.values()))
    print(dict_test, sum(dict_test.values()))