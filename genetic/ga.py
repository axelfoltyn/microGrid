import random

import sys, os

import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.abspath('')))


from microGrid.env.final_env import MyEnv as MG_two_storages_env
from microGrid.callback.callback import BestCallback, ResetCallback

from parameters import EnvParam, Defaults

from stable_baselines3 import DQN

from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.utils import set_random_seed


def random_pop(min_val, max_val, nb_val, nb_pop, gener_rnd, is_int = True):
    """initial population, the population is in the form of a list of values.
        Args:
            min_val: minimum value for the list
            max_val: maximum value for the list
            nb_val: number of values in the list
            nb_pop: number of values in list
            gener_rnd: create by numpy.random.default_rng() before
                methode need is random and integers
            is_int: boolean to chose if each value is an integer of a float
        Return:
            the list of generated populations
    """
    if is_int:
        return [[gener_rnd.integers(min_val, max_val) for _ in range(nb_val)] for _ in range(nb_pop)]
    return [[gener_rnd.random() * (max_val - min_val) + min_val for _ in range(nb_val)] for _ in range(nb_pop)]


def crossover(pop, r_cross, nb_ind, gener_rnd):
    """ create two new individuals by crossing
        Args:
            pop: populations selected for mutation
            nb_ind: number of individuals to generate
            r_cross: percentage for there to be a crossover
            gener_rnd: create by numpy.random.default_rng() before
                methode need is random and integers
        Return:
            list [c1, c2], the two new individuals
    """
    res = []
    for _ in range(nb_ind//2):
        # children are copies of parents by default
        p1 = pop[gener_rnd.integers(0, len(pop) - 1)].copy()
        p2 = pop[gener_rnd.integers(0, len(pop) - 1)].copy()
        c1, c2 = p1, p2
        # check for recombination
        if gener_rnd.random() < r_cross:
            # select crossover point that is not on the end of the string
            pt = gener_rnd.integers(0, len(p1)-1)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        res.append(c1)
        res.append(c2)
    return res

def mutation(min_val, max_val, pop, nb_ind, r_mut, gener_rnd, is_int = True):
    """ create a new individual by mutation
            Args:
                min_val: minimum value
                max_val: maximum value
                pop: populations selected for mutation
                nb_ind: number of individuals to generate
                r_mut: percentage for there to be a change on a value.
                gener_rnd: create by numpy.random.default_rng() before
                    methode need is random and integers
                is_int: boolean to chose if each value is an integer of a float
            Return:
                the new deferent individual
    """
    res = []
    for _ in range(nb_ind):
        child = pop[gener_rnd.integers(0, len(pop) - 1)].copy()
        change = False
        while not change:
            for i in range(len(child)):
                if gener_rnd.random() < r_mut:
                    # change value
                    tmp = child[i]
                    if is_int:
                        child[i] = gener_rnd.integers(min_val, max_val)
                    else:
                        child[i] = gener_rnd.random() * (max_val - min_val) + min_val
                    if tmp != child[i]:
                        change = True
        res.append(child)
    return res


def selection(pop, scores, gener_rnd, nb_ind, k=3):
    """ create a new individual by mutation
        Args:
            min_val: minimum value
            max_val: maximum value
            pop: populations selected for mutation
            nb_ind: number of individuals selected
            r_mut: percentage for there to be a change on a value.
            gener_rnd: create by numpy.random.default_rng() before
                methode need is random and integers
            k: number of tries before taking the best
        Return:
            the new individual
    """
    # first random selection
    selected = []
    no_select = list(range(len(pop)))
    for _ in range(nb_ind):
        if len(no_select) > 0:
            s = gener_rnd.integers(len(no_select))
            for i in gener_rnd.integers(0, len(no_select), k-1):
                # check if better (e.g. perform a tournament)
                if no_select[s] not in selected and scores[no_select[i]] < scores[no_select[s]]:
                    s = i
            selected.append(no_select.pop(s))

        else:
            selected.append(gener_rnd.integers(len(pop)))
    return [pop[i] for i in selected]


def eval(dirname, filename, l_coeff, env_test, lfn, lreset, lname, val=0, patience = 15):
    """
    :param dirname: folder in which the files of the best individuals are temporarily stored.
    :param filename: beginning of the file names of the best individuals.
    :param l_coeff: list of coefficients for each training sub-function.
    :param env_test: list of test environments.
    :param lreset: list of test functions to be reset before calculation.
    :param val: seed to fix random function
    :param patience: time out with no better result.
    :return: list of scores obtained for each test environment.
    """
    res = []
    lenvs = create_env(l_coeff, lfn, lname)

    for envs in lenvs:
        res2 = []
        env = envs[0]
        env_valid = envs[1]
        # create a DQN agent with a fixed seed random
        set_random_seed(val)

        model = DQN('MlpPolicy', env,
                    buffer_size=10 ** 4,
                    batch_size=2 ** 8,
                    gamma=0.8,
                    exploration_initial_eps=Defaults.EPSILON_START,
                    exploration_final_eps=Defaults.EPSILON_MIN,
                    exploration_fraction=4 * 10 ** (-6),
                    seed=val,
                    verbose=0)

        best = BestCallback(env_valid, {}, patience, filename, dirname, verbose=False, save_all=False)
        reset = ResetCallback(lreset)


        model.learn(Defaults.EPISODE * Defaults.STEPS_PER_EPISODE,
                    callback=[reset, best, reset])

        model.load(best.get_best_name())
        for env in env_test:
            for elt in lreset:
                elt.reset()
            tmp = evaluate_policy(model, env, n_eval_episodes=1, return_episode_rewards=True)
            tmp = [i / j for i, j in zip(tmp[0], tmp[1])]
            res2.append(np.mean(tmp))
        res.append(res2)
        try:
            os.remove(best.get_best_name()) # not need model
        except OSError as err:
            print(best.get_best_name(), "not exist to remove\n",err)
    print("res", res)
    return np.mean(res, axis=0).tolist() # mean column



def create_env(l_coeff, lfn, lname):
    """ create a new individual by mutation
        Args:
           l_coeff: coefficients for each reward function
           lfn: list of reward function used
           lname: name for each reward function
        Return: [[connected_env, connected_env_valid]]
           list of environment list (train env + valid env).
    """

    connected_env = MG_two_storages_env(np.random.RandomState(),
                              pred=EnvParam.PREDICTION,
                              dist_equinox=EnvParam.EQUINOX,
                              length_history=EnvParam.LENGTH_HISTORY,
                              max_ener_buy=None,
                              max_ener_sell=None)

    connected_env_valid = MG_two_storages_env(np.random.RandomState(),
                              pred=EnvParam.PREDICTION,
                              dist_equinox=EnvParam.EQUINOX,
                              length_history=EnvParam.LENGTH_HISTORY,
                              max_ener_buy=None,
                              max_ener_sell=None)

    for i in range(len(lfn)):
        connected_env.add_reward(lname[i], lambda x, fn=lfn[i], coeff=l_coeff[i]: fn(x) * coeff)
        connected_env_valid.add_reward(lname[i], lambda x, fn=lfn[i], coeff=l_coeff[i]: fn(x) * coeff)

    return [[connected_env, connected_env_valid]]

