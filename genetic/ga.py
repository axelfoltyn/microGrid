import random

import sys, os

import numpy as np

sys.path.append(os.path.abspath(os.path.abspath('')))
print(os.path.abspath(os.path.abspath('')))


from microGrid.env.final_env import MyEnv as MG_two_storages_env
from microGrid.callback.callback import BestCallback, ResetCallback
from microGrid.reward.reward import BlackoutReward

from parameters import EnvParam, Defaults

from stable_baselines3 import DQN

from stable_baselines3.common.evaluation import evaluate_policy

from datetime import datetime


def random_pop(min_val, max_val, nb_val, nb_pop, gener_rnd):
    """initial population, the population is in the form of a list of values.
        Args:
            min_val: minimum value for the list
            max_val: maximum value for the list
            nb_val: number of values in the list
            nb_pop: number of values in list
            gener_rnd: create by numpy.random.default_rng() before
                methode need is random
        Return:
            the list of generated populations
    """
    return [[gener_rnd.random() * (max_val - min_val) + min_val for _ in range(nb_val)] for _ in range(nb_pop)]


def crossover(pop, r_cross, nb_ind, gener_rnd):
    """ create two new individuals by crossing
        Args:
            pop: populations selected for mutation
            nb_ind:
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

def mutation(min_val, max_val, pop, nb_ind, r_mut, gener_rnd):
    """ create a new individual by mutation
            Args:
                min_val
                max_val
                pop
                r_mut: percentage for there to be a change on a value.
                gener_rnd: create by numpy.random.default_rng() before
                    methode need is random
            Return:
                the new individual
        """
    res = []
    for _ in range(nb_ind):
        child = pop[gener_rnd.integers(0, len(pop) - 1)].copy()
        for i in range(len(child)):
            if gener_rnd.random() < r_mut:
                # change value
                child[i] = gener_rnd.random() * (max_val - min_val) + min_val
        res.append(child)
    return res

def create_env(l_coeff):
    lres_reset = []
    reward_blackout = BlackoutReward()
    lres_reset.append(reward_blackout)
    reward_valid_blackout = BlackoutReward()
    lres_reset.append(reward_valid_blackout)
    max_blackout = 365. * 24


    connected_env = MG_two_storages_env(np.random.RandomState(),
                              pred=EnvParam.PREDICTION,
                              dist_equinox=EnvParam.EQUINOX,
                              length_history=EnvParam.LENGTH_HISTORY,
                              max_ener_buy=None,
                              max_ener_sell=None)
    connected_env.add_reward("Waste", lambda x: (1-x["waste_energy"]) * l_coeff[0])
    connected_env.add_reward("Blackout", lambda x: (1. - reward_blackout.fn(x) / max_blackout) * l_coeff[1])
    connected_env.add_reward("Profit_buy", lambda x: (1-x["buy_energy"]) * l_coeff[2])
    connected_env.add_reward("Profit_sell", lambda x: (x["sell_energy"]) * l_coeff[3])

    no_connect_env = MG_two_storages_env(np.random.RandomState(),
                               pred=EnvParam.PREDICTION,
                               dist_equinox=EnvParam.EQUINOX,
                               length_history=EnvParam.LENGTH_HISTORY,
                               max_ener_buy=0,
                               max_ener_sell=0)

    no_connect_env.add_reward("Waste", lambda x: (1-x["waste_energy"]) * l_coeff[0])
    no_connect_env.add_reward("Blackout", lambda x: (1. - reward_blackout.fn(x) / max_blackout) * l_coeff[1])
    no_connect_env.add_reward("Profit_buy", lambda x: (1-x["buy_energy"]) * l_coeff[2])
    no_connect_env.add_reward("Profit_sell", lambda x: (x["sell_energy"]) * l_coeff[3])

    connected_env_valid = MG_two_storages_env(np.random.RandomState(),
                              pred=EnvParam.PREDICTION,
                              dist_equinox=EnvParam.EQUINOX,
                              length_history=EnvParam.LENGTH_HISTORY,
                              max_ener_buy=None,
                              max_ener_sell=None)
    connected_env_valid.add_reward("Waste", lambda x: (1-x["waste_energy"]) * l_coeff[0])
    connected_env_valid.add_reward("Blackout", lambda x: (1. - reward_blackout.fn(x) / max_blackout) * l_coeff[1])
    connected_env_valid.add_reward("Profit_buy", lambda x: (1-x["buy_energy"]) * l_coeff[2])
    connected_env_valid.add_reward("Profit_sell", lambda x: (x["sell_energy"]) * l_coeff[3])

    no_connect_env_valid = MG_two_storages_env(np.random.RandomState(),
                               pred=EnvParam.PREDICTION,
                               dist_equinox=EnvParam.EQUINOX,
                               length_history=EnvParam.LENGTH_HISTORY,
                               max_ener_buy=0,
                               max_ener_sell=0)

    no_connect_env_valid.add_reward("Waste", lambda x: (1-x["waste_energy"]) * l_coeff[0])
    no_connect_env_valid.add_reward("Blackout", lambda x: (1. - reward_blackout.fn(x) / max_blackout) * l_coeff[1])
    no_connect_env_valid.add_reward("Profit_buy", lambda x: (1-x["buy_energy"]) * l_coeff[2])
    no_connect_env_valid.add_reward("Profit_sell", lambda x: (x["sell_energy"]) * l_coeff[3])

    return [[connected_env, connected_env_valid], [no_connect_env, no_connect_env_valid]], lres_reset

def create_env_test():
    lres_reset = []
    lenv = []
    reward_blackout = BlackoutReward()
    lres_reset.append(reward_blackout)
    max_blackout = 365. * 24


    lenv.append(MG_two_storages_env(np.random.RandomState(),
                              pred=EnvParam.PREDICTION,
                              dist_equinox=EnvParam.EQUINOX,
                              length_history=EnvParam.LENGTH_HISTORY,
                              max_ener_buy=0,
                              max_ener_sell=0))
    lenv[-1].add_reward("Waste", lambda x: (1-x["waste_energy"]))

    lenv.append(MG_two_storages_env(np.random.RandomState(),
                               pred=EnvParam.PREDICTION,
                               dist_equinox=EnvParam.EQUINOX,
                               length_history=EnvParam.LENGTH_HISTORY,
                               max_ener_buy=0,
                               max_ener_sell=0))
    lenv[-1].add_reward("Blackout", lambda x: (1. - reward_blackout.fn(x) / max_blackout))

    lenv.append(MG_two_storages_env(np.random.RandomState(),
                              pred=EnvParam.PREDICTION,
                              dist_equinox=EnvParam.EQUINOX,
                              length_history=EnvParam.LENGTH_HISTORY,
                              max_ener_buy=None,
                              max_ener_sell=None))

    lenv[-1].add_reward("Profit_buy", lambda x: (1-x["buy_energy"]))

    lenv.append(MG_two_storages_env(np.random.RandomState(),
                               pred=EnvParam.PREDICTION,
                               dist_equinox=EnvParam.EQUINOX,
                               length_history=EnvParam.LENGTH_HISTORY,
                               max_ener_buy=0,
                               max_ener_sell=0))

    lenv[-1].add_reward("Profit_sell", lambda x: (x["sell_energy"]))

    return lenv, lres_reset


def selection(pop, scores, gener_rnd, k=3):
    # first random selection
    selection_ix = gener_rnd.integers(len(pop))
    for ix in gener_rnd.integers(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def eval(dirname, filename, l_coeff, env_test, lreset_test, val=0, patience = 15):
    res = []
    lenvs, lreset = create_env(l_coeff)
    for envs in lenvs:
        env = envs[0]
        env_valid = envs[1]
        # create a DQN agent with a fixed seed random
        random.seed(val)
        model = DQN('MlpPolicy', env,
                    learning_rate=1e-5,
                    buffer_size=10**5,
                    batch_size=2**8,
                    gamma=0.8,
                    exploration_initial_eps=Defaults.EPSILON_START,
                    exploration_final_eps=Defaults.EPSILON_MIN,
                    exploration_fraction=4 * 10**(-6),
                    target_update_interval=50,
                    train_freq=2,
                    tau=0.95,
                    seed=val,
                    verbose=0)

        best = BestCallback(env_valid, {}, patience, filename, dirname)
        reset = ResetCallback(lreset)


        model.learn(Defaults.EPISODE * Defaults.STEPS_PER_EPISODE,
                    callback=[reset, best, reset])

        model.load(best.get_best_name())
        for env in env_test: # TODO faire proprement env (vu que env par pop)
            tmp = evaluate_policy(model, env, return_episode_rewards=True)
            print(tmp)
            res.append(np.mean([t[0] / t[1] for t in tmp]))
        #res = [evaluate_policy(model, env)[0] for env in env_test]
        for elt in lreset_test:
            elt.reset()
    return [np.mean(r) for r in res]


if __name__ == "__main__":
    import time
    start = time.time()
    N=10 # Nombre d’itérations avant arrêt de l’algorithme
    G = 10 # Nombre de générations avant de commencer les mutations
    nb_ind = 5 # Nombre de population à chaque iteration
    r_mut = 0.5
    r_cross = 0.5
    # creer la moitier en mutation et l'autre en crossover
    nb_mut = nb_ind % 2 + nb_ind // 2 - (nb_ind // 2) % 2
    nb_cross = nb_ind // 2 + (nb_ind // 2) % 2
    val = 0
    mu = 5 #nb_parent selected to create new child
    rnd = np.random.default_rng() #random.randint(0, sys.maxsize)
    env_test, lreset = create_env_test()

    pop = random_pop(-10, 10, len(env_test), nb_ind, rnd)
    print(pop)
    scores= [0 for _ in pop]

    for nb_iter in range(N):
        if nb_iter < G:
            pop = random_pop(-10, 10, len(env_test), nb_ind, rnd)
        else:
            pop_select = [selection(pop, [sum(s) for s in scores], rnd) for _ in range(mu)]
            pop = mutation(-10, 10, pop_select, nb_mut, r_mut, rnd) + crossover(pop_select, r_cross, nb_cross, rnd)
        #eval
        scores = [eval("eval", str(p), p, env_test, lreset, val=val, patience = 15) for p in pop]
        print("scores", scores)
        #insert map-elites grid and change score with nolety value
        print("pop", pop)
    res = time.time() - start
    print("time to train and valid:", int(res / 3600), "h", int((res % 3600) / 60), "min", res % 60, "s")