import gc
import logging

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import sys
import time

import tensorflow as tf
import pandas as pd
from stable_baselines3 import DQN

from sklearn import preprocessing
from microGrid.reward.reward import ClientReward, BlackoutReward, DODReward, DOD2Reward, Client2Reward

from datetime import datetime
import shutil

from microGrid.tools.tool import Rainflow

print(os.path.abspath(''))

sys.path.append(os.path.abspath(os.path.abspath('')))
print(os.path.abspath(os.path.abspath('')))


from microGrid.env.final_env import MyEnv as MG_two_storages_env
from microGrid.callback.callback import BestCallback, ResetCallback
from microGrid.plot_MG_operation import plot_op



class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPISODE = 365 * 24 - 1
    EPISODE = 60
    STEPS_PER_TEST = 365 * 24 - 1


    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 1e-5
    LEARNING_RATE_DECAY = 0.99
    DISCOUNT = 0.8
    DISCOUNT_INC = 0.99
    DISCOUNT_MAX = 0.98
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 4e-6
    REPLAY_MEMORY_SIZE = 100000
    BATCH_SIZE = 256
    FREEZE_INTERVAL = 50
    DETERMINISTIC = False
    TARGET_UPDATE_INTERVAL = 2


# parameters used in the initialization of environments
class EnvParam:
    # The max value of the network purchase (None = no limit)
    MAX_BUY_ENERGY = None
    # The maximum value of the network sale (None = no limit)
    MAX_SELL_ENERGY = None
    # To know the production and consumption of the following
    PREDICTION = False
    # To have the deadline before the next solstice
    EQUINOX = True
    # Size of the production/consumption history
    LENGTH_HISTORY = 12

def init_env(connect):
    rng = np.random.RandomState()
    env_res = dict()
    env_valid_res = dict()
    absolute_dir = os.path.abspath('')
    prod = np.load(absolute_dir + "/microGrid/env/data/BelgiumPV_prod_test.npy")[0:1 * 365 * 24]
    cons = np.load(absolute_dir + "/microGrid/env/data/example_nondeterminist_cons_test.npy")[0:1 * 365 * 24]
    max_buy = EnvParam.MAX_BUY_ENERGY
    max_sell = EnvParam.MAX_SELL_ENERGY
    if not connect:
        max_buy = 0
        max_sell = 0

    # --- Instantiate reward parameters ---
    price_h2 = 0.1  # 0.1euro/kWh of hydrogen
    price_elec_buy = 2.0  # 2euro/kWh
    cost_wast = 0.1  # arbitrary value
    
    # Fixme : il faut reset pour l'automatisation
    lres_reset = []
    reward_client = ClientReward()
    lres_reset.append(reward_client)
    reward_client_valid = ClientReward()
    lres_reset.append(reward_client_valid)
    reward_client2 = Client2Reward()
    lres_reset.append(reward_client2)
    reward_client2_valid = Client2Reward()
    lres_reset.append(reward_client2_valid)
    reward_blackout = BlackoutReward()
    lres_reset.append(reward_blackout)
    reward_valid_blackout = BlackoutReward()
    lres_reset.append(reward_valid_blackout)
    dod_reward = DODReward(Rainflow())
    lres_reset.append(dod_reward)
    dod_reward_valid = DODReward(Rainflow())
    lres_reset.append(dod_reward_valid)
    dod2_reward = DOD2Reward()
    lres_reset.append(dod2_reward)
    dod2_reward_valid = DOD2Reward()
    lres_reset.append(dod2_reward_valid)



    if connect:
        # profit réseau
        key = "vente"
        env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                           length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                           max_ener_sell=max_sell)

        env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                                 pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                                 length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                                 max_ener_sell=max_sell)
        env_res[key].add_reward("Profit", lambda x: (x["sell_energy"]) * price_elec_buy, 1.)

        env_valid_res[key].add_reward("Profit", lambda x: (x["sell_energy"]) * price_elec_buy, 1.)

        # profit réseau
        key = "(-prix_achat)_x_achat"
        env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                           length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                           max_ener_sell=max_sell)

        env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                                 pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                                 length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                                 max_ener_sell=max_sell)
        env_res[key].add_reward("Profit", lambda x: (-x["buy_energy"]) * price_elec_buy, 1.)
        env_valid_res[key].add_reward("Profit", lambda x: (-x["buy_energy"]) * price_elec_buy, 1.)


    if not connect:
        key = "(-cout_coupure)_x_coupure"

        env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                           length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                           max_ener_sell=max_sell)

        env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                                 pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                                 length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                                 max_ener_sell=max_sell)

        env_res[key].add_reward("Blackout", lambda x: reward_blackout.fn(x), 1.)
        env_valid_res[key].add_reward("Blackout", lambda x: reward_valid_blackout.fn(x), 1.)

        # ressenti client
        key = "f_insatisfait(temps_sans_energie)"
        env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                           length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                           max_ener_sell=max_sell)

        env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                                 pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                                 length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                                 max_ener_sell=max_sell)
        env_res[key].add_reward("Dissatisfaction", lambda x: reward_client.fn(x), 1.)
        env_valid_res[key].add_reward("Dissatisfaction", lambda x: reward_client_valid.fn(x), 1.)

        # optimisation énergie
        key = "(-perte_d’energie)_x_cout_perte"
        env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                           length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                           max_ener_sell=max_sell)

        env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                                 pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                                 length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                                 max_ener_sell=max_sell)
        env_res[key].add_reward("Waste", lambda x: -x["waste_energy"] * cost_wast, 1.)
        env_valid_res[key].add_reward("Waste", lambda x: -x["waste_energy"] * cost_wast, 1.)

    key = "flux_batterie_h2"

    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                       max_ener_sell=max_sell)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                             max_ener_sell=max_sell)

    env_res[key].add_reward("Flow_H2", lambda x: x["flow_H2"], 1.)
    env_valid_res[key].add_reward("Flow_H2", lambda x: x["flow_H2"], 1.)

    key = "flux_batterie_lithium"

    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                       max_ener_sell=max_sell)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                             max_ener_sell=max_sell)

    env_res[key].add_reward("Flow_lithium", lambda x: x["flow_lithium"], 1.)
    env_valid_res[key].add_reward("Flow_lithium", lambda x: x["flow_lithium"], 1.)

    key = "dod"

    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                       max_ener_sell=max_sell)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                             max_ener_sell=max_sell)

    env_res[key].add_reward("Dod", lambda x: dod_reward.fn(x), 1.)
    env_valid_res[key].add_reward("Dod", lambda x: dod_reward_valid.fn(x), 1.)

    key = "dod2"

    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                       max_ener_sell=max_sell)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                             max_ener_sell=max_sell)

    env_res[key].add_reward("Dod", lambda x: -dod2_reward.fn(x), 1.)
    env_valid_res[key].add_reward("Dod", lambda x: -dod2_reward_valid.fn(x), 1.)

    key = "client2"
    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                       max_ener_sell=max_sell)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                             max_ener_sell=max_sell)
    env_res[key].add_reward("Dissatisfaction", lambda x: reward_client2.fn(x), 1.)
    env_valid_res[key].add_reward("Dissatisfaction", lambda x: reward_client2_valid.fn(x), 1.)

    return env_res, env_valid_res, lres_reset

def del_env(dict_env):
    for env in dict_env.values():
        del(env)
    del(dict_env)
    gc.collect()

def reset_all(lreset):
    for elt in lreset:
        elt.reset()




def main(env_res=None, env_valid_res=None, lreset=None):
    start = time.time()
    patience = None
    dirname = "result_corr_norm_test"
    # --- Instantiate environments ---
    if env_res is None or env_valid_res is None or lreset is None:
        env_res, env_valid_res, lreset  = init_env(True)
    for connect in [True, False]:
        # del env to prevent MemoryError
        del_env(env_res)
        del_env(env_valid_res)
        # --- Instantiate environments ---
        env_res, env_valid_res, lreset = init_env(connect)
        print("tensorflow work with:", tf.test.gpu_device_name())
        logging.basicConfig(level=logging.INFO)

        pow_lr = 5
        decay = 4 * 10**(-6)
        pow_replaybuff_size = 5
        tau = 0.95
        pow_buff_size = 8
        discount = 0.8
        train_freq = 2
        freeze = 50


        for key in env_res.keys():

            env = env_res[key]
            env_valid = env_valid_res[key]
            dict_env = {k:val for (k,val) in env_res.items() if k != key}
            now = datetime.now()
            # dd_mm_YY-H-M-S
            dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")
            filename = "best" + dt_string + "_env_" + key + "_connected_" + str(connect)
            test(dirname, filename,
                 patience,
                 train_freq,
                 learning_rate= 10**(-pow_lr),
                 buffer_size= 10**(pow_replaybuff_size),
                 batch_size = 2**(pow_buff_size),
                 discount= discount,
                 eps_decay= decay,
                 freeze = freeze,
                 dict_env=dict_env,
                 env=env,
                 env_valid=env_valid,
                 tau=tau,
                 verbose=False, param=str(key), lreset=lreset)
    res = time.time() - start
    print("tot time:", int(res / 3600), "h", int((res % 3600) / 60), "min", res % 60, "s")


def test(dirname, filename,
         patience,
         train_freq,
         learning_rate,
         buffer_size,
         batch_size,
         discount,
         eps_decay,
         freeze,
         dict_env,
         env,
         env_valid,
         tau=1.,
         lreset=[],
         verbose = False, param=""):





    # --- init model ---
    print('MlpPolicy',
          "learning_rate=", learning_rate,
          "buffer_size=", buffer_size,
          "batch_size=", batch_size,
          "gamma=", discount,
          "exploration_initial_eps=", Defaults.EPSILON_START,
          "exploration_final_eps=", Defaults.EPSILON_MIN,
          "exploration_fraction=", eps_decay,
          "target_update_interval=", freeze,
          "size_histo=", EnvParam.LENGTH_HISTORY,
          "train_freq=", train_freq,
          "tau=", tau, sep='\n')
    if not os.path.exists(dirname + "/" + filename):
        os.makedirs(dirname + "/" + filename)

    f = open(dirname+ "/" + filename + "/" + filename + "hyperparam.txt", "a", encoding="utf-8")
    f.write('MlpPolicy\n' +
          "learning_rate=" +  str(learning_rate) +
          "\nbuffer_size=" +  str(buffer_size) +
          "\nbatch_size=" +  str(batch_size) +
          "\ngamma=" +  str(discount) +
          "\nexploration_initial_eps=" +  str(Defaults.EPSILON_START) +
          "\nexploration_final_eps=" +  str(Defaults.EPSILON_MIN) +
          "\nexploration_fraction=" +  str(eps_decay) +
          "\ntarget_update_interval=" +  str(freeze) +
          "\ntrain_freq=" + str(train_freq) +
          "\nsize_histo=" + str(EnvParam.LENGTH_HISTORY) +
          "\nPREDICTION=" + str(EnvParam.PREDICTION) +
          "\nEQUINOX=" + str(EnvParam.EQUINOX) +
          "\ntau=" + str(tau) +
          "\nparam=" + str(param)
    )
    f.close()

    print(len(env.observation_space.sample()))

    model = DQN('MlpPolicy', env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                batch_size=batch_size,
                gamma=discount,
                exploration_initial_eps=Defaults.EPSILON_START,
                exploration_final_eps=Defaults.EPSILON_MIN,
                exploration_fraction=eps_decay,
                target_update_interval=freeze,
                train_freq=train_freq,
                tau=tau,
                verbose=0)

    best = BestCallback(env_valid, dict_env, patience, filename, dirname)
    reset = ResetCallback(lreset)

    try:
        start = time.time()
        model.learn(Defaults.EPISODE * Defaults.STEPS_PER_EPISODE,
                    callback=[reset, best, reset])  # callback=[verbose_callback, eps_callback, best_callback]
        res = time.time() - start
        print("time to train and valid:", int(res / 3600), "h", int((res % 3600) / 60), "min", res % 60, "s")
    except KeyboardInterrupt:
        print('Hello user you have KeyboardInterrupt.')
    plot_gene(best, dirname, filename, verbose=verbose)


def plot_gene(best, dirname, filename, verbose=False, param=""):
    if not verbose:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")

    data = best.get_data()
    bestScores, allScores = best.get_score()
    #if we haven't enouth point we can't ananlisis reward
    if len(bestScores[list(bestScores.keys())[0]]) < 0:
        shutil.rmtree(dirname + "/" + filename, ignore_errors=True)
        return
    print(data.keys())

    actions = data["action"]
    consumption = data["consumption"]
    production = data["production"]
    rewards = data["rewards"]
    battery_level = data["soc"]
    # plot_op(data["action"], data["consumption"], data["production"], data["rewards"], data["battery"], "test.png")
    i = 0
    plot_op(actions[0 + i:100 + i], consumption[0 + i:100 + i], production[0 + i:100 + i], rewards[0 + i:100 + i],
            battery_level[0 + i:100 + i], dirname + "/" + filename + "/" + filename + "_winter_.png")
    plt.title("winter")
    if verbose:
        plt.show()
    plt.clf()

    i = 180 * 24
    plot_op(actions[0 + i:100 + i], consumption[0 + i:100 + i], production[0 + i:100 + i], rewards[0 + i:100 + i],
            battery_level[0 + i:100 + i], dirname + "/" + filename + "/" + filename + "_summer_.png")
    plt.title("summer")
    if verbose:
        plt.show()
    plt.clf()

    i = 360 * 24
    plot_op(actions[0 + i:100 + i], consumption[0 + i:100 + i], production[0 + i:100 + i], rewards[0 + i:100 + i],
            battery_level[0 + i:100 + i], dirname + "/" + filename + "/" + filename + "_winter2_.png")
    plt.title("winter2")
    if verbose:
        plt.show()
    plt.clf()

    i = 0
    lines = []
    key = "battery_h2"
    p, = plt.plot(range(len(data[key][i:i + int(365 / 3 * 24)])), data[key][i:i + int(365 / 3 * 24)], label=key,
                  color='b', alpha=0.5)
    lines.append(p)
    ax = plt.gca()
    ax2 = ax.twinx()
    key = "soc"
    p, = ax2.plot(range(len(data[key][i:i + int(365 / 3 * 24)])), data[key][i:i + int(365 / 3 * 24)], label=key,
                  color='r', alpha=0.5)
    lines.append(p)
    plt.legend(lines, [l.get_label() for l in lines])


    plt.xlabel("Number of hours")
    plt.ylabel("Score")
    plt.savefig(dirname + "/" + filename + "/" + filename + "_plots1.png")
    if verbose:
        plt.show()
    plt.clf()

    lines = []
    i += int(365 / 3 * 24)
    key = "battery_h2"
    p, = plt.plot(range(len(data[key][i:i + int(365 / 3 * 24)])), data[key][i:i + int(365 / 3 * 24)], label=key, color='b', alpha=0.5)
    lines.append(p)
    ax = plt.gca()
    ax2 = ax.twinx()
    key = "soc"
    p, = ax2.plot(range(len(data[key][i:i + int(365 / 3 * 24)])), data[key][i:i + int(365 / 3 * 24)], label=key, color='r', alpha=0.5)
    lines.append(p)
    plt.legend(lines, [l.get_label() for l in lines])
    plt.xlabel("Number of hours")
    plt.ylabel("Score")
    plt.savefig(dirname + "/" + filename + "/" + filename + "_plots2.png")
    if verbose:
        plt.show()
    plt.clf()

    lines = []
    i += int(365 / 3 * 24)
    key = "battery_h2"
    p, = plt.plot(range(len(data[key][i:i + int(365 / 3 * 24)])), data[key][i:i + int(365 / 3 * 24)], label=key, color='b', alpha=0.5)
    lines.append(p)
    ax = plt.gca()
    ax2 = ax.twinx()
    key = "soc"
    p, = ax2.plot(range(len(data[key][i:i + int(365 / 3 * 24)])), data[key][i:i + int(365 / 3 * 24)], label=key,
                  color='r', alpha=0.5)
    lines.append(p)
    ax2.plot(range(len(data[key][i:i + int(365 / 3 * 24)])), data[key][i:i + int(365 / 3 * 24)], label=key, color='r', alpha=0.5)

    plt.legend(lines, [l.get_label() for l in lines])
    plt.xlabel("Number of hours")
    plt.ylabel("Score")
    plt.savefig(dirname + "/" + filename + "/" + filename + "_plots3.png")
    if verbose:
        plt.show()
    plt.clf()



    h = sns.jointplot(x=[battery_level[i] for i in range(len(actions)) if actions[i] == 0],
                      y=[consumption[i] - production[i] for i in range(len(actions)) if actions[i] == 0],
                      kind="hist", marginal_ticks=True)
    # JointGrid has a convenience function
    h.set_axis_labels('charge battery (%)', 'demand (W)', fontsize=16)
    h.fig.suptitle("distribution selon l'action de décharge")

    plt.savefig(dirname + "/" + filename + "/" + filename + "_plots_action0.png", bbox_inches = "tight")
    if verbose:
        plt.show()
    plt.clf()

    h = sns.jointplot(x=[battery_level[i] for i in range(len(actions)) if actions[i] == 1],
                      y=[consumption[i] - production[i] for i in range(len(actions)) if actions[i] == 1],
                      kind="hist", marginal_ticks=True)
    # JointGrid has a convenience function
    h.set_axis_labels('charge battery (%)', 'demand (W)', fontsize=16)
    h.fig.suptitle("distribution selon l'action ne rien faire")
    plt.savefig(dirname + "/" + filename + "/" + filename + "_plots_action1.png", bbox_inches = "tight")
    if verbose:
        plt.show()
    plt.clf()

    h = sns.jointplot(x=[battery_level[i] for i in range(len(actions)) if actions[i] == 2],
                      y=[consumption[i] - production[i] for i in range(len(actions)) if actions[i] == 2],
                      kind="hist", marginal_ticks=True)
    # JointGrid has a convenience function
    h.set_axis_labels('charge battery (%)', 'demand (W)', fontsize=16)
    h.fig.suptitle("distribution selon l'action de charge")
    plt.savefig(dirname + "/" + filename + "/" + filename + "_plots_action2.png", bbox_inches = "tight")
    if verbose:
        plt.show()
    plt.clf()

    demande = [consumption[i] - production[i] for i in range(len(actions))]
    print("demande moyenne : ", np.mean(demande))
    print("demande std : ", np.std(demande))

    corr = pd.DataFrame.from_dict(allScores)
    corr = corr.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True)
    plt.title("correlation entre les différentes valeurs de récompense")
    plt.savefig(dirname + "/" + filename + "/" + filename + "_heatmap.png", bbox_inches = "tight")
    if verbose:
        plt.show()
    plt.clf()


    corr = pd.DataFrame.from_dict(data)
    corr = corr.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True)
    plt.title("correlation entre tous les données")
    plt.savefig(dirname + "/" + filename + "/" + filename + "_heatmap2.png", bbox_inches = "tight")
    if verbose:
        plt.show()
    plt.clf()
    print("reward", np.sum(data["rewards"]))

    scaler = preprocessing.MinMaxScaler()

    for k in bestScores.keys():
        try:
            plt.plot(range(1,len(bestScores[k])+1), scaler.fit_transform(np.array(bestScores[k]).reshape(-1,1)),
                     label="score " + str(k))
        except Exception as e:
            print("error",k)
            print(e)
            plt.plot(range(1,len(bestScores[k])+1), bestScores[k], label="score " + str(k))

    plt.legend()
    plt.xlabel("pas pour chaque nouveaux meilleurs scores (sans unité)")
    plt.ylabel("score normalisé (sans unité)")
    plt.title("courbe d'évolution normalisé des meilleur scores")
    plt.savefig(dirname + "/" + filename + "/" + filename + "_scoresbest.png", bbox_inches = "tight")
    if verbose:
        plt.show()
    plt.clf()

    for k in allScores.keys():
        try:
            plt.plot(range(1, len(allScores[k])+1), allScores[k],
                     label="score " + str(k))
        except Exception as e:
            print("error",k)
            print(e)
            plt.plot(range(1, len(allScores[k])+1), allScores[k], label="score " + str(k))

    plt.legend()
    plt.xlabel("nombre d'épisode (sans unité)")
    plt.ylabel("score (sans unité)")
    plt.title("courbe d'évolution des scores pour chaque episode d'apprentissage")
    plt.savefig(dirname + "/" + filename + "/" + filename + "_scores.png", bbox_inches = "tight")
    if verbose:
        plt.show()
    plt.clf()

    for k in allScores.keys():
        try:
            plt.plot(range(1, len(allScores[k])+1), scaler.fit_transform(np.array(allScores[k]).reshape(-1,1)),
                     label="score " + str(k))
        except Exception as e:
            print("error",k)
            print(e)
            plt.plot(range(1, len(allScores[k])+1), scaler.fit_transform(np.array(allScores[k]).reshape(-1,1)), label="score " + str(k))

    plt.legend()
    plt.xlabel("nombre d'épisode (sans unité)")
    plt.ylabel("score normalisé (sans unité)")
    plt.title("courbe d'évolution normalsé des scores pour chaque episode d'apprentissage")
    plt.savefig(dirname + "/" + filename + "/" + filename + "_scores_normed.png", bbox_inches = "tight")
    if verbose:
        plt.show()

    plt.clf()

    labels = 'discharge', 'none', 'charge'
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    i=0
    sizes = [len([1 for j in range(len(actions[i:i+int(365 / 3 * 24)])) if actions[i+j] == 0]),
             len([1 for j in range(len(actions[i:i+int(365 / 3 * 24)])) if actions[i+j] == 1]),
             len([1 for j in range(len(actions[i:i+int(365 / 3 * 24)])) if actions[i+j] == 2])]


    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)

    plt.axis('equal')
    plt.title("camembert des proportion des actions choisies")

    plt.savefig(dirname + "/" + filename + "/" + filename + '_PieChart1.png', bbox_inches = "tight")
    if verbose:
        plt.show()
    plt.clf()

    i+= int(365 / 3 * 24)
    sizes = [len([1 for j in range(len(actions[i:i + int(365 / 3 * 24)])) if actions[i + j] == 0]),
             len([1 for j in range(len(actions[i:i + int(365 / 3 * 24)])) if actions[i + j] == 1]),
             len([1 for j in range(len(actions[i:i + int(365 / 3 * 24)])) if actions[i + j] == 2])]

    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)

    plt.axis('equal')
    plt.title("camembert des proportion des actions choisies")

    plt.savefig(dirname + "/" + filename + "/" + filename + '_PieChart2.png', bbox_inches="tight")
    if verbose:
        plt.show()
    plt.clf()

    i += int(365 / 3 * 24)
    sizes = [len([1 for j in range(len(actions[i:i + int(365 / 3 * 24)])) if actions[i + j] == 0]),
             len([1 for j in range(len(actions[i:i + int(365 / 3 * 24)])) if actions[i + j] == 1]),
             len([1 for j in range(len(actions[i:i + int(365 / 3 * 24)])) if actions[i + j] == 2])]

    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)

    plt.axis('equal')
    plt.title("camembert des proportion des actions choisies")

    plt.savefig(dirname + "/" + filename + "/" + filename + '_PieChart3.png', bbox_inches="tight")
    if verbose:
        plt.show()
    plt.clf()



    plt.close("all")

    with open(dirname + "/" + filename + "/" + filename + "_data.csv", 'w', encoding="utf-8") as f:
        keys = list(allScores.keys())
        for j in range(len(keys) - 1):
            f.write(str(keys[j]) + ";")
        f.write(str(keys[-1]) + "\n")
        print(allScores)
        print(keys[0], ":len", len(allScores[keys[0]]))
        for i in range(len(allScores[keys[0]])): # same size in allScores
            for j in range(len(keys) - 1):
                f.write(str(allScores[keys[j]][i]) + ";")
            f.write(str(allScores[keys[-1]][i]) + "\n")



def init_env2():
    rng = np.random.RandomState()
    env_res = dict()
    env_valid_res = dict()
    absolute_dir = os.path.abspath('')
    prod = np.load(absolute_dir + "/microGrid/env/data/BelgiumPV_prod_test.npy")[0:1 * 365 * 24]
    cons = np.load(absolute_dir + "/microGrid/env/data/example_nondeterminist_cons_test.npy")[0:1 * 365 * 24]
    max_buy = EnvParam.MAX_BUY_ENERGY
    max_sell = EnvParam.MAX_SELL_ENERGY

    # --- Instantiate reward parameters ---

    lres_reset = []
    reward_blackout = BlackoutReward()
    lres_reset.append(reward_blackout)
    reward_valid_blackout = BlackoutReward()
    lres_reset.append(reward_valid_blackout)

    # profit réseau
    key = "vente"
    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                       max_ener_sell=max_sell)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                             max_ener_sell=max_sell)
    env_res[key].add_reward("Profit", lambda x: (x["sell_energy"]), 1.)

    env_valid_res[key].add_reward("Profit", lambda x: (x["sell_energy"]), 1.)

    # profit réseau
    key = "achat"
    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                       max_ener_sell=max_sell)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=max_buy,
                                             max_ener_sell=max_sell)
    env_res[key].add_reward("Profit", lambda x: (-x["buy_energy"]), 1.)
    env_valid_res[key].add_reward("Profit", lambda x: (-x["buy_energy"]), 1.)


    key = "(-cout_coupure)_x_coupure"
    max_blackout = 365. * 24

    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=0,
                                       max_ener_sell=0)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=0,
                                             max_ener_sell=0)

    env_res[key].add_reward("Blackout", lambda x: reward_blackout.fn(x) / max_blackout, 1.)
    env_valid_res[key].add_reward("Blackout", lambda x: reward_valid_blackout.fn(x) / max_blackout, 1.)



    # optimisation énergie
    key = "(-perte_d’energie)_x_cout_perte"
    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=0,
                                       max_ener_sell=0)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=0,
                                             max_ener_sell=0)
    env_res[key].add_reward("Waste", lambda x: -x["waste_energy"], 1.)
    env_valid_res[key].add_reward("Waste", lambda x: -x["waste_energy"], 1.)



    return env_res, env_valid_res, lres_reset

if __name__ == "__main__":

    main()
