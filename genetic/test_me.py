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
    lenv[-1].add_reward("Blackout", lambda x: (1. + reward_blackout.fn(x) / max_blackout))

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


if __name__ == "__main__":
    import time
    start = time.time()
    N=20 # Nombre d’itérations avant arrêt de l’algorithme
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
    map_fn, map_score, map_cut = creat_map([20 for _ in env_test], 0, 1)

    pop = random_pop(-10, 10, len(env_test), nb_ind, rnd)
    print(pop)
    scores = [[0] for _ in pop]
    min_val = 0
    max_val = 10
    coor_set = set()
    for nb_iter in range(N):
        if nb_iter < G:
            pop = random_pop(min_val, max_val, len(env_test), nb_ind, rnd)
        else:
            pop_select = [selection(pop, [sum(s) for s in scores], rnd) for _ in range(mu)]
            pop = mutation(min_val, max_val, pop_select, nb_mut, r_mut, rnd) + crossover(pop_select, r_cross, nb_cross, rnd)
        #eval
        scores = [eval("eval", str(p), p, env_test, lreset, val=val, patience = 15) for p in pop]
        print("scores", scores)

        coor = [coor_map(map_cut, score) for score in scores]
        print("coordone map", coor)
        #insert map-elites grid and insert score with novelty value
        for i in range(len(pop)):
            scores[i].append(insert_map(map_fn, map_score, coor[i], pop[i], scores[i]))
        coor_set.union(set(coor))
        print("pop", pop)
        print("scores", scores)
    print("coor", coor_set, lent(coor_set))
    res = time.time() - start
    print("time to train and valid:", int(res / 3600), "h", int((res % 3600) / 60), "min", res % 60, "s")