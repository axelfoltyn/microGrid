from gym.spaces import Discrete, Box, Dict, Tuple
import gym
import numpy as np
import os

"""
modified Vincent Francois-Lavet code
"""
class MicroGrid (gym.Env):
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
    def __init__(self, reduce_qty_data=None, length_history=None, start_history=None,production_norm=None,
                 consumption_norm=None, scale_prod=12000. / 1000 / 2, scale_cons=2.1, random_start=True):

        reduce_qty_data = int(reduce_qty_data) if reduce_qty_data is not None else int(1)
        length_history = int(length_history) if length_history is not None else int(12)
        start_history = int(start_history) if start_history is not None else int(0)
        print("reduce_qty_data, length_history, start_history")
        print(reduce_qty_data, length_history, start_history)
        # Defining the type of environment
        self._dist_equinox = 0
        self._pred = 0
        self._reduce_qty_data = reduce_qty_data  # Factor by which to artificially reduce the data available (for training+validation)
        # Choices are 1,2,4,8,16

        self._length_history = length_history  # Length for the truncature of the history to build the pseudo-state

        self._start_history = start_history  # Choice between data that is replicated (choices are in [0,...,self._reduce_qty_data[ )

        inc_sizing = 1.

        if (self._dist_equinox == 1 and self._pred == 1):
            self._last_ponctual_observation = [0., [0., 0.], 0., [0., 0.]]
            self._input_dimensions = [(1,), (self._length_history, 2), (1,), (1, 2)]
        elif (self._dist_equinox == 1 and self._pred == 0):
            self._last_ponctual_observation = [0., [0., 0.], 0.]
            self._input_dimensions = [(1,), (self._length_history, 2), (1,)]
        elif (self._dist_equinox == 0 and self._pred == 0):
            self._last_ponctual_observation = [0., [0., 0.]]
            self._input_dimensions = [(1,), (self._length_history, 2)]

        absolute_dir = os.path.dirname(os.path.abspath(__file__))
        if consumption_norm is None:
            consumption_norm = np.load(absolute_dir + "/data/example_nondeterminist_cons_train.npy")[
                               0:1 * 365 * 24]
        if production_norm is None:
            production_norm = np.load(absolute_dir + "/data/BelgiumPV_prod_train.npy")[0:1 * 365 * 24]
        self.production_norm = production_norm
        self.consumption_norm = consumption_norm
        self.consumption = self.consumption_norm * scale_cons
        self.production = self.production_norm * scale_prod

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

        if (self._reduce_qty_data != 1):
            for season in range(4):
                self.production_train[season * nd_one_seas:(season + 1) * nd_one_seas] = np.tile(
                    self.production_train[
                                     int((season + (
                                                 self._start_history + 0.) / self._reduce_qty_data) * nd_one_seas):int(
                                         (season + (
                                                     self._start_history + 1.) / self._reduce_qty_data) * nd_one_seas)
                    ], self._reduce_qty_data)
                self.production_valid[season * nd_one_seas:(season + 1) * nd_one_seas] = np.tile(
                    self.production_valid[
                                     int((season + (
                                                 self._start_history + 0.) / self._reduce_qty_data) * nd_one_seas):int(
                                         (season + (
                                                     self._start_history + 1.) / self._reduce_qty_data) * nd_one_seas)
                    ], self._reduce_qty_data)

                self.production_train_norm[season * nd_one_seas:(season + 1) * nd_one_seas] = np.tile(
                    self.production_train_norm[
                    int((season + (self._start_history + 0.) / self._reduce_qty_data) * nd_one_seas):int(
                        (season + (self._start_history + 1.) / self._reduce_qty_data) * nd_one_seas)],
                    self._reduce_qty_data)
                self.production_valid_norm[season * nd_one_seas:(season + 1) * nd_one_seas] = np.tile(
                    self.production_valid_norm[
                    int((season + (self._start_history + 0.) / self._reduce_qty_data) * nd_one_seas):int(
                        (season + (self._start_history + 1.) / self._reduce_qty_data) * nd_one_seas)],
                    self._reduce_qty_data)

        # dict_reward stores the reward function
        # dict_coeff stores the reward coefficient
        # (used in add_reward and my_reward)
        self.dict_reward = dict()
        self.dict_coeff = dict()
        #init parameter dictionary
        self._init_dict()

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

    def inputDimensions(self):
        return self._input_dimensions

    def _init_obs(self):
        # part deerv2
        if (self._dist_equinox == 1 and self._pred == 1):
            self._last_ponctual_observation = [
                0.,
                [0., 0.],
                0.,
                [0., 0.]
            ]
            self._input_dimensions = [(1,), (self._length_history, 2), (1,), (1, 2)]
        elif (self._dist_equinox == 1 and self._pred == 0):
            self._last_ponctual_observation = [
                0.,
                [0., 0.],
                0.
            ]
            self._input_dimensions = [(1,), (self._length_history, 2), (1,)]
        elif (self._dist_equinox == 0 and self._pred == 0):
            self._last_ponctual_observation = [
                0.,
                [0., 0.]
            ]
            self._input_dimensions = [(1,), (self._length_history, 2)]
        # part openai.gym
        self.observation_space = Tuple([
            Box(low=np.zeros(shape), high=np.ones(shape), dtype=np.float64)
            for shape in self._input_dimensions])

    def step(self, action):
        reward = 0  # self.ale.act(action)  #FIXME
        terminal = 0

        true_demand = self.consumption[self.counter - 1] - self.production[self.counter - 1]

        diff_hydrogen = 0
        # action 0 discharges H2 reserve  1 do nothing and 2 charge H2 reserve
        if (action == 0):
            ## Energy is taken out of the hydrogen reserve
            true_energy_avail_from_hydrogen = -self.hydrogen_max_power * self.hydrogen_eta
            diff_hydrogen = -self.hydrogen_max_power
        if (action == 1):
            ## No energy is taken out of/into the hydrogen reserve
            true_energy_avail_from_hydrogen = 0
            diff_hydrogen = 0
        if (action == 2):
            ## Energy is taken into the hydrogen reserve
            true_energy_avail_from_hydrogen = self.hydrogen_max_power / self.hydrogen_eta
            diff_hydrogen = self.hydrogen_max_power



        self.dict_param["flow_H2"] = diff_hydrogen
        self.hydrogen_storage += diff_hydrogen

        Energy_needed_from_battery = true_demand + true_energy_avail_from_hydrogen

        if (Energy_needed_from_battery > 0):
            # Lack of energy
            if (self._last_ponctual_observation[0] * self.battery_size > Energy_needed_from_battery):
                # If enough energy in the battery, use it
                self.dict_param["lack_energy"] = 0
                self._last_ponctual_observation[0] = self._last_ponctual_observation[0] \
                                                     - Energy_needed_from_battery / self.battery_size / self.battery_eta
            else:
                # Otherwise: use what is left and then penalty
                self.dict_param["lack_energy"] = (Energy_needed_from_battery - self._last_ponctual_observation[
                    0] * self.battery_size)
                self._last_ponctual_observation[0] = 0
        elif (Energy_needed_from_battery < 0):
            # Surplus of energy --> load the battery
            self.dict_param["waste_energy"] = max(0, (self._last_ponctual_observation[0] * self.battery_size
                                             - Energy_needed_from_battery * self.battery_eta) - self.battery_size)
            self._last_ponctual_observation[0] = min(1., self._last_ponctual_observation[0]
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
        self._last_ponctual_observation[1][0] = self.consumption_norm[self.counter]
        self._last_ponctual_observation[1][1] = self.production_norm[self.counter]
        i = 1
        if (self._dist_equinox == 1):
            i += 1
            self._last_ponctual_observation[i] = abs(self.counter / 24 - (365. / 2)) / (
                        365. / 2)  # 171 days between 1jan and 21 Jun
        if (self._pred == 1):
            i += 1
            self._last_ponctual_observation[i][0] = sum(
                self.production_norm[self.counter:self.counter + 24]) / 24.  # *self.rng.uniform(0.75,1.25)
            self._last_ponctual_observation[i][1] = sum(
                self.production_norm[self.counter:self.counter + 48]) / 48.  # *self.rng.uniform(0.75,1.25)
        self.counter += 1

        dict_reward = self.my_reward()
        if self._pred == 1:
            done = self.counter + 24 >= len(self.production_norm)
        else:
            done = self.counter >= len(self.production_norm)
        info = {}
        # print("===================>", self._last_ponctual_observation)
        return self._last_ponctual_observation, np.sum(list(dict_reward.values())), done, info

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
        return self._last_ponctual_observation

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