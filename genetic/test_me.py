import csv

from matplotlib import pyplot as plt
from scipy.spatial import KDTree

from ga import random_pop, selection, mutation, crossover, eval
from mapelites import creat_map, coor_map, insert_map, get_d

import numpy as np

import  sys, os
sys.path.append(os.path.abspath(os.path.abspath('')))
print(os.path.abspath(os.path.abspath('')))


from microGrid.env.final_env import MyEnv as MG_two_storages_env
from microGrid.reward.reward import BlackoutReward, CountBuyReward

from parameters import EnvParam

def create_env_testV0():
    """
    :return: list of name environement, list of environement and list of function need to reset.
    """
    lres_reset = []
    lenv = []
    lname = []
    reward_blackout = BlackoutReward()
    lres_reset.append(reward_blackout)
    max_blackout = 365. * 24

    lname.append("Waste")
    lenv.append(MG_two_storages_env(np.random.RandomState(),
                              pred=EnvParam.PREDICTION,
                              dist_equinox=EnvParam.EQUINOX,
                              length_history=EnvParam.LENGTH_HISTORY,
                              max_ener_buy=0,
                              max_ener_sell=0))
    lenv[-1].add_reward(lname[-1], lambda x: (1-x["waste_energy"]))

    lname.append("Blackout")
    lenv.append(MG_two_storages_env(np.random.RandomState(),
                               pred=EnvParam.PREDICTION,
                               dist_equinox=EnvParam.EQUINOX,
                               length_history=EnvParam.LENGTH_HISTORY,
                               max_ener_buy=0,
                               max_ener_sell=0))
    lenv[-1].add_reward(lname[-1], lambda x: (1. + reward_blackout.fn(x) / max_blackout))

    lname.append("Profit_buy")
    lenv.append(MG_two_storages_env(np.random.RandomState(),
                              pred=EnvParam.PREDICTION,
                              dist_equinox=EnvParam.EQUINOX,
                              length_history=EnvParam.LENGTH_HISTORY,
                              max_ener_buy=None,
                              max_ener_sell=None))

    lenv[-1].add_reward(lname[-1], lambda x: (1-x["buy_energy"]))

    lname.append("Profit_sell")
    lenv.append(MG_two_storages_env(np.random.RandomState(),
                               pred=EnvParam.PREDICTION,
                               dist_equinox=EnvParam.EQUINOX,
                               length_history=EnvParam.LENGTH_HISTORY,
                               max_ener_buy=None,
                               max_ener_sell=None))

    lenv[-1].add_reward(lname[-1], lambda x: (x["sell_energy"]))

    return lname, lenv, lres_reset

def create_env_test(lfn, lname):
    """
    :return: list of name environement, list of environement and list of function need to reset.
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

def create_env_testV1():
    """
    :return: list of name environement, list of environement and list of function need to reset.
    """
    max_buy = 365. * 24
    lres_reset = []
    reward_count_buy = CountBuyReward()
    lres_reset.append(reward_count_buy)
    reward_valid_count_buy = CountBuyReward()
    lres_reset.append(reward_valid_count_buy)
    lenv = []
    lenv.append(MG_two_storages_env(np.random.RandomState(),
                                    pred=EnvParam.PREDICTION,
                                    dist_equinox=EnvParam.EQUINOX,
                                    length_history=EnvParam.LENGTH_HISTORY,
                                    max_ener_buy=None,
                                    max_ener_sell=None))
    lenv[-1].add_reward("Number_buy", lambda x: (1. + reward_count_buy.fn(x) / max_buy))

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
                                    max_ener_buy=None,
                                    max_ener_sell=None))
    lenv[-1].add_reward("Profit_sell", lambda x: (x["sell_energy"]))

    return lenv, lres_reset

def creat_lfn():
    lres_reset = []
    lfn = []
    lname =[]
    lcut = []
    nb_cut = 30
    max_buy = 365. * 24

    reward_cunt_buy = CountBuyReward()
    lres_reset.append(reward_cunt_buy)
    reward_valid_count_buy = CountBuyReward()
    lres_reset.append(reward_valid_count_buy)

    """lname.append("Number_buy")
    lfn.append(lambda x: (1. + reward_cunt_buy.fn(x) / max_buy))
    lcut.append([1 - 1 / (2*i) for i in range(1, nb_cut + 1)])"""
    lname.append("Profit_buy")
    lfn.append(lambda x: (1 - x["buy_energy"]))
    lcut.append([1 - 1 / (2*i) for i in range(1, nb_cut + 1)])
    lname.append("Profit_sell")
    lfn.append(lambda x: (x["sell_energy"]))
    lcut.append([1 / (2*i) for i in range(nb_cut, 0, -1)])
    return lfn, lres_reset, lname, lcut

if __name__ == "__main__":
    import time
    start = time.time()

    ############################
    #  values has initialized  #
    ############################
    min_val = 0
    max_val = 50
    seed_val = 0 # seed to fix random in eval


    N = 10 # Number of iterations before stopping the algorithm
    G = 5 # Number of generations before starting mutations (initial pop)

    nb_ind = 5 # Number of population generated at each iteration
    r_mut = 0.4 # probability that an element of an individual changes
    r_cross = 1. # probability that there is a crossover

    # create half in mutation and half in crossover
    nb_mut = nb_ind % 2 + nb_ind // 2 - (nb_ind // 2) % 2
    nb_cross = nb_ind // 2 + (nb_ind // 2) % 2

    mu = 5 # number of parents selected to create new children

    nb_fit = mu // 2 # number of parents selected by fitness score
    nb_novelty = mu - nb_fit # number of parents selected by novelty score

    k = 10 # K-nearest neighbors to calculate the novelty score

    rnd = np.random.default_rng() #random function need beacous the seed of random is fixed in eval

    lfn, lreset, lname, map_cut = creat_lfn()

    #names, env_test, lreset = create_env_test()
    #env_test, lreset_test = create_env_test()
    env_test= create_env_test(lfn, lname)

    # dictionary and set that represents the map (key: tuple of coordinate value: ind or score save)
    coor_set = set()
    dict_map = dict()
    dict_map_s = dict()
    # use for novelty score
    lcoor = []

    ############################
    #  start of the agorithm  #
    ############################
    ## initial population and calculation of its score (novelty + fitness)
    pop = random_pop(min_val, max_val, len(env_test), G, rnd)
    print(pop)
    scores = [eval("eval", str(p), p, env_test, lfn, lreset, lname, val=seed_val, patience = 15) for p in pop]
    print(scores)

    # collecting the coordinates of each individual
    coor = list([coor_map(map_cut, score) for score in scores])
    lcoor += coor
    # save new coordinate in a set
    coor_set.update([tuple(l) for l in coor])

    # calculation of novelty score (distance)
    kd_tree = KDTree(np.array(lcoor))
    dist = [get_d(kd_tree, c, min(len(coor_set), k)) for c in coor]
    print("dist", dist, type(dist))

    # add the difference between the individual's score and the one stored in the map
    for i in range(len(scores)):
        scores[i].append(insert_map(dict_map, dict_map_s, coor[i], pop[i], scores[i]))

    print("pop", pop)
    print("scores", scores)
    print("coor", coor_set, len(coor_set))


    for nb_iter in range(N):
        # selects the parents on which a mutation will be made
        pop_select = selection(pop, [sum(s) for s in scores], rnd, nb_fit) + selection(pop, dist, rnd, nb_novelty)
        print("pop_select", pop_select)
        pop = mutation(min_val, max_val, pop_select, nb_mut, r_mut, rnd) + crossover(pop_select, r_cross, nb_cross, rnd)
        print("pop", pop, type(pop))
        #eval
        scores = [eval("eval", str(p), p, env_test, lfn, lreset, lname, val=seed_val, patience = 15) for p in pop]



        coor = list([coor_map(map_cut, score) for score in scores])
        lcoor += coor
        #kd_tree = KDTree(np.array(list(map(list, coor_set))))
        kd_tree = KDTree(np.array(lcoor))
        dist = [get_d(kd_tree, c, min(len(coor_set), k)) for c in coor]
        print("dist", dist, type(dist))
        #insert map-elites grid and insert score with novelty value
        for i in range(len(scores)):
            scores[i].append(insert_map(dict_map, dict_map_s, coor[i], pop[i], scores[i]))
        coor_set.update([tuple(l) for l in coor])
        print("scores", scores, type(scores))
        print("coor", coor_set, len(coor_set))
    print("coor", coor_set, len(coor_set))

    with open('test.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        # write the header
        header = lname
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


    res = time.time() - start
    print("time to train and valid:", int(res / 3600), "h", int((res % 3600) / 60), "min", res % 60, "s")