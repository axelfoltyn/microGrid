import csv
import shutil

from scipy.spatial import KDTree
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from ga import random_pop, selection, mutation, crossover, eval
from mapelites import coor_map, insert_map, get_d

import numpy as np

import sys, os

from microGrid.callback.callback import BestCallback, ResetCallback

sys.path.append(os.path.abspath(os.path.abspath('')))
print(os.path.abspath(os.path.abspath('')))


from microGrid.env.final_env import MyEnv as MG_two_storages_env
from microGrid.reward.reward import BlackoutReward, CountBuyReward

from parameters import EnvParam, Defaults


def create_env_test(lfn, lname):
    """

    :param lfn: list of function used
    :param lname: list of name for each function

    :return: list of name environment, list of environment and list of function need to reset.
    """
    lenv = []
    for i in range(len(lfn)):
        lenv.append(MG_two_storages_env(np.random.RandomState(),
                                        pred=EnvParam.PREDICTION,
                                        dist_equinox=EnvParam.EQUINOX,
                                        length_history=EnvParam.LENGTH_HISTORY,
                                        max_ener_buy=None,
                                        max_ener_sell=None))
        lenv[-1].add_reward(lname[i], lambda x, fn=lfn[i]: fn(x))
    return lenv

def creat_lfn():
    """
    :return: lfn, lres_reset, lname, lcut
        lfn: list of function used
        lres_reset: list of function need to reset each start episode
        lname: list of name for each function
        lcut: cutting value lists of our map
    """

    def _set_fn(fn, lreset, is_neg, born_min, born_max, nb_cut=20):
        """

        :param fn: the lambda function
        :param lreset: list of reset function
        :param is_neg: true if all solutions must be negative, false otherwise
        :param born_min and born_max: the terminals used for cutting
        :param nb_cut: how much cutting needs to be done to delimit the set
        :return:
            maximum and minimum absolute values
        """
        def _get_mean(fn, lreset, val=0):
            tmp_env = MG_two_storages_env(np.random.RandomState(),
                                          pred=EnvParam.PREDICTION,
                                          dist_equinox=EnvParam.EQUINOX,
                                          length_history=EnvParam.LENGTH_HISTORY,
                                          max_ener_buy=None,
                                          max_ener_sell=None)
            tmp_env.add_reward("Test", lambda x, f=fn: abs(fn(x)))

            tmp_model = DQN('MlpPolicy', tmp_env,
                            buffer_size=10 ** 4,
                            batch_size=2 ** 8,
                            gamma=0.8,
                            exploration_initial_eps=Defaults.EPSILON_START,
                            exploration_final_eps=Defaults.EPSILON_MIN,
                            exploration_fraction=4 * 10 ** (-6),
                            seed=val,
                            verbose=0)

            best = BestCallback(tmp_env, {}, 10, "suppr", "suppr", verbose=False, save_all=False)
            reset = ResetCallback(lreset)

            tmp_model.learn(Defaults.EPISODE * Defaults.STEPS_PER_EPISODE,
                            callback=[reset, best, reset])

            os.remove(best.get_best_name())  # not need model
            shutil.rmtree("suppr")
            tmp = evaluate_policy(tmp_model, tmp_env, n_eval_episodes=1, return_episode_rewards=True)
            tmp = [i / j for i, j in zip(tmp[0], tmp[1])]

            max_res = np.mean(tmp)

            tmp_env = MG_two_storages_env(np.random.RandomState(),
                                          pred=EnvParam.PREDICTION,
                                          dist_equinox=EnvParam.EQUINOX,
                                          length_history=EnvParam.LENGTH_HISTORY,
                                          max_ener_buy=None,
                                          max_ener_sell=None)
            tmp_env.add_reward("Test", lambda x, f=fn: -abs(fn(x)))

            tmp_model = DQN('MlpPolicy', tmp_env,
                            buffer_size=10 ** 4,
                            batch_size=2 ** 8,
                            gamma=0.8,
                            exploration_initial_eps=Defaults.EPSILON_START,
                            exploration_final_eps=Defaults.EPSILON_MIN,
                            exploration_fraction=4 * 10 ** (-6),
                            seed=val,
                            verbose=0)

            best = BestCallback(tmp_env, {}, 10, "suppr", "suppr", verbose=False, save_all=False)
            reset = ResetCallback(lreset)

            tmp_model.learn(Defaults.EPISODE * Defaults.STEPS_PER_EPISODE,
                            callback=[reset, best, reset])

            try:
                os.remove(best.get_best_name())  # not need model
            except OSError as err:
                print(best.get_best_name(), "not exist to remove\n", err)
            shutil.rmtree("suppr", ignore_errors=True)
            tmp = evaluate_policy(tmp_model, tmp_env, n_eval_episodes=1, return_episode_rewards=True)
            tmp = [i / j for i, j in zip(tmp[0], tmp[1])]

            min_res = abs(np.mean(tmp))

            print(max_res, min_res)
            return max_res, min_res

        if is_neg:
            cut = [(born_max + born_min) * (1 - 1 / nb_cut * i) for i in range(1, nb_cut + 1)]
        else:
            cut = [(born_max + born_min) / nb_cut * i for i in range(1, nb_cut + 1)]
        max_mean, min_mean = _get_mean(fn, lreset)
        if max_mean != min_mean:
            if is_neg:
                return (lambda x, f=fn, mean_max=max_mean, mean_min=min_mean: (f(x) + mean_max) / (
                        mean_max - mean_min) - 1), cut
            return (lambda x, f=fn, mean_max=max_mean, mean_min=min_mean: (f(x) - mean_min) / (
                    mean_max - mean_min)), cut
        else:
            return (lambda x, f=fn, mean_max=max_mean: f(x) / mean_max), cut
    lres_reset = []
    lfn = []
    lname =[]
    lcut = []

    nb_cut = 20

    reward_cunt_buy = CountBuyReward()
    lres_reset.append(reward_cunt_buy)
    reward_valid_count_buy = CountBuyReward()
    lres_reset.append(reward_valid_count_buy)

    lname.append("Number_buy")
    fn, cut = _set_fn(lambda x: reward_cunt_buy.fn(x), lres_reset, True, -1, 0, nb_cut)
    lfn.append(fn)
    lcut.append(cut)


    lname.append("Profit_buy")
    fn, cut = _set_fn(lambda x: -x["buy_energy"], lres_reset, True, -1, 0, nb_cut)
    lfn.append(fn)
    lcut.append(cut)


    lname.append("Waste")
    fn, cut = _set_fn(lambda x: -x["sell_energy"], lres_reset, True, -1, 0, nb_cut)
    lfn.append(fn)
    lcut.append(cut)

    return lfn, lres_reset, lname, lcut

def write_data(lname, coor_set, dict_map, dict_map_s, name):
    print([s[-1] > 0.1 for s in scores])


    with open(name + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        # write the header
        header = lname.copy()
        header += ["formule", "sum_score"] + ["scores " + str(name) for name in header]
        writer.writerow(header)

        # write the data
        for c in coor_set:
            line = list(c)
            line += [dict_map[c]]
            s = dict_map_s[c]
            line += [sum(s)]
            line += s
            writer.writerow(line)
    print(name + '.csv', "created")

if __name__ == "__main__":
    import time
    start = time.time()

    ############################
    #  values has initialized  #
    ############################
    min_val = 0
    max_val = 50
    seed_val = 3 # seed to fix random in eval
    name = "test" # file name to save data
    k = 10 # K-nearest neighbors to calculate the novelty score
    t = 0.1 # threshold before counting an update of our data


    N = 10 # patience before stopping the algorithm
    G = 5 # Number of generations before starting mutations (initial pop)

    ## genetic values
    nb_ind = 5 # Number of population generated at each iteration
    r_mut = 0.4 # probability that an element of an individual changes
    r_cross = 1. # probability that there is a crossover
    mu = 5 # number of parents selected to create new children

    # create half in mutation and half in crossover
    nb_mut = nb_ind % 2 + nb_ind // 2 - (nb_ind // 2) % 2
    nb_cross = nb_ind // 2 + (nb_ind // 2) % 2

    # number of parents selected by fitness score
    nb_fit = mu // 2
    # number of parents selected by novelty score
    nb_novelty = mu - nb_fit

    ############################
    #  start of the agorithm   #
    ############################
    # random function need because the seed of random is fixed in eval
    rnd = np.random.default_rng()

    # create reward function and environement
    lfn, lreset, lname, map_cut = creat_lfn()
    env_test = create_env_test(lfn, lname)

    # dictionary and set that represents the map (key: tuple of coordinate value: ind or score save)
    coor_set = set()
    dict_map = dict()
    dict_map_s = dict()
    # use for novelty score
    lcoor = []

    ## initial population and calculation of its score (novelty + fitness)
    pop = random_pop(min_val, max_val, len(env_test), G, rnd)
    scores = [eval("eval", str(p), p, env_test, lfn, lreset, lname, val=seed_val, patience = 15) for p in pop]

    # collecting the coordinates of each individual
    coor = list([coor_map(map_cut, score) for score in scores])
    lcoor += [s.copy() for s in scores]
    # save new coordinate in a set
    coor_set.update([tuple(l) for l in coor])

    # calculation of novelty score (distance)
    kd_tree = KDTree(np.array(lcoor))
    dist = [get_d(kd_tree, s, min(len(lcoor), k)) for s in scores]

    # add the difference between the individual's score and the one stored in the map
    for i in range(len(scores)):
        scores[i].append(insert_map(dict_map, dict_map_s, coor[i], pop[i], scores[i]))

    cpt2 = 0
    nb_point = len(coor_set)
    try:
        cpt = 0
        while cpt <= N:
            cpt +=1
            # selects the parents on which a mutation will be made
            pop_select = selection(pop, [sum(s) for s in scores], rnd, nb_fit) + selection(pop, dist, rnd, nb_novelty)
            pop = mutation(min_val, max_val, pop_select, nb_mut, r_mut, rnd) + crossover(pop_select, r_cross, nb_cross, rnd)
            #eval
            scores = [eval("eval", str(p), p, env_test, lfn, lreset, lname, val=seed_val, patience = 15) for p in pop]

            coor = list([coor_map(map_cut, score) for score in scores])
            lcoor += [s.copy() for s in scores]
            kd_tree = KDTree(np.array(lcoor))
            dist = [get_d(kd_tree, s, min(len(lcoor), k)) for s in scores]
            #insert map-elites grid and insert score with novelty value
            for i in range(len(scores)):
                scores[i].append(insert_map(dict_map, dict_map_s, coor[i], pop[i], scores[i]))
            # if there is an improvement, reset the patience counter
            print([s[-1] > t for s in scores])
            print(len(coor_set), " > ", nb_point)
            if any([s[-1] > t for s in scores]) or len(coor_set) > nb_point:
                cpt = 0
                cpt2 += 1
                print(len(coor_set), " > ", nb_point)
                write_data(lname, coor_set, dict_map, dict_map_s, name + "_" + str(cpt2))
                nb_point = len(coor_set)
            coor_set.update([tuple(l) for l in coor])
    except KeyboardInterrupt:
        print('Hello user you have KeyboardInterrupt.')
    print("coor", coor_set, len(coor_set))

    write_data(lname, coor_set, nb_point, dict_map, dict_map_s, name + "_final_seed_" + str(seed_val))


    res = time.time() - start
    print("time to train and valid:", int(res / 3600), "h", int((res % 3600) / 60), "min", res % 60, "s")