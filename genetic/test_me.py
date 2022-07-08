import csv

from matplotlib import pyplot as plt

from ga import random_pop, selection, mutation, crossover, eval
from mapelites import creat_map, coor_map, insert_map

import numpy as np

import  sys, os
sys.path.append(os.path.abspath(os.path.abspath('')))
print(os.path.abspath(os.path.abspath('')))


from microGrid.env.final_env import MyEnv as MG_two_storages_env
from microGrid.callback.callback import BestCallback, ResetCallback
from microGrid.reward.reward import BlackoutReward

from parameters import EnvParam

def create_env_test():
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


if __name__ == "__main__":
    import time
    start = time.time()

    min_val = 0
    max_val = 10

    N = 10 # Nombre d’itérations avant arrêt de l’algorithme
    G = 2 # Nombre de générations avant de commencer les mutations

    nb_ind = 5 # Nombre de population à chaque iteration
    r_mut = 0.4
    r_cross = 1.
    # creer la moitier en mutation et l'autre en crossover
    nb_mut = nb_ind % 2 + nb_ind // 2 - (nb_ind // 2) % 2
    nb_cross = nb_ind // 2 + (nb_ind // 2) % 2

    mu = 5 #nb_parent selected to create new child
    rnd = np.random.default_rng() #random.randint(0, sys.maxsize)

    names, env_test, lreset = create_env_test()

    seed_val = 0
    dict_map = dict()
    dict_map_s = dict()
    #map_fn, map_score, map_cut = creat_map([20 for _ in env_test], 0, 1)
    _, _, map_cut = creat_map([20 for _ in env_test], 0, 1)
    coor_set = set()



    pop = random_pop(min_val, max_val, len(env_test), G, rnd)
    print(pop)
    scores = [eval("eval", str(p), p, env_test, lreset, val=seed_val, patience = 15) for p in pop]
    print(pop)
    coor = list([coor_map(map_cut, score) for score in scores])
    coor_set.update([tuple(l) for l in coor])
    for i in range(len(scores)):
        # scores[i].append(insert_map(map_fn, map_score, coor[i], pop[i], scores[i]))
        scores[i].append(insert_map(dict_map, dict_map_s, coor[i], pop[i], scores[i]))

    with open('test.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        # write the header
        header = names
        header += ["formule"] + ["scores " + str(name) for name in header]
        writer.writerow(header)

        # write the data
        for c in coor_set:
            line = list(c) # todo a modif
            line += [dict_map[c]]
            s = dict_map_s[c]
            line += [sum(s)]
            line += s
            writer.writerow(line)

    print("pop", pop)
    print("scores", scores)
    print("coor", coor_set, len(coor_set))

    for nb_iter in range(N):
        """if nb_iter < G:
            pop = list(random_pop(min_val, max_val, len(env_test), nb_ind, rnd))
        else:"""
        pop_select = selection(pop, [sum(s) for s in scores], rnd, mu)
        print("pop_select", pop_select)
        pop = mutation(min_val, max_val, pop_select, nb_mut, r_mut, rnd) + crossover(pop_select, r_cross, nb_cross, rnd)
        print("pop", pop, type(pop))
        #eval
        scores = [eval("eval", str(p), p, env_test, lreset, val=seed_val, patience = 15) for p in pop]
        print("scores", scores, type(scores))

        coor = list([coor_map(map_cut, score) for score in scores])
        print("coordone", coor, type(coor))
        #insert map-elites grid and insert score with novelty value
        for i in range(len(scores)):
            #scores[i].append(insert_map(map_fn, map_score, coor[i], pop[i], scores[i]))
            scores[i].append(insert_map(dict_map, dict_map_s, coor[i], pop[i], scores[i]))
        #TODO faire mieux (set of tuple ?)
        """for coor_elem in coor:
            if all([not (l == coor_elem) for l in coor_set]):
                coor_set.append(coor_elem)"""
        coor_set.update([tuple(l) for l in coor])
        print("pop", pop)
        print("scores", scores)
        print("coor", coor_set, len(coor_set))
    print("coor", coor_set, len(coor_set))

    with open('test.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        # write the header
        header = names
        header += ["formule"] + ["scores " + str(name) for name in header]
        writer.writerow(header)

        # write the data
        for c in coor_set:
            line = list(c) # todo a modif
            line += [dict_map[c]]
            s = dict_map_s[c]
            line += [sum(s)]
            line += s
            writer.writerow(line)

    # TODO faire
    #fig, ax = plt.subplots()
    #x_label_list = ['A2', 'B2', 'C2', 'D2']
    #ax.set_xticks(range(0, 10))

    #ax.set_xticklabels(x_label_list)
    #ax.imshow(z, extent=[-1, 1, -1, 1])


    res = time.time() - start
    print("time to train and valid:", int(res / 3600), "h", int((res % 3600) / 60), "min", res % 60, "s")