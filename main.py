import logging

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import time

import tensorflow as tf
import pandas as pd
from stable_baselines3 import DQN

from sklearn import preprocessing
from microGrid.reward.reward import ClientReward

from datetime import datetime

print(os.path.abspath(''))

sys.path.append(os.path.abspath(os.path.abspath('')))
print(os.path.abspath(os.path.abspath('')))


from microGrid.env.final_env import MyEnv as MG_two_storages_env
from microGrid.callback.callback import BestCallback
from microGrid.plot_MG_operation import plot_op



class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 365 * 24 - 1
    EPOCHS = 200
    STEPS_PER_TEST = 365 * 24 - 1
    PERIOD_BTW_SUMMARY_PERFS = -1  # Set to -1 for avoiding call to env.summarizePerformance

    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 1

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.02
    LEARNING_RATE_DECAY = 0.99
    DISCOUNT = 0.99
    DISCOUNT_INC = 0.99
    DISCOUNT_MAX = 0.98
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_NORM = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 2.3e-5
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 200#32
    FREEZE_INTERVAL = 1000
    DETERMINISTIC = False
    TARGET_UPDATE_INTERVAL = 2

class EnvParam:
    MAX_BUY_ENERGY = None
    MAX_SELL_ENERGY = 0
    PREDICTION = False
    EQUINOX = True
    LENGTH_HISTORY = 1

def main():
    now = datetime.now()
    # dd_mm_YY-H-M-S
    dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")
    dirname = "result"
    filename = "best" + dt_string
    patience = 20

    print("tensorflow work with:", tf.test.gpu_device_name())
    logging.basicConfig(level=logging.INFO)
    rng = np.random.RandomState()

    # --- Instantiate environment ---
    dict_env = dict()

    env = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                              length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                              max_ener_sell=EnvParam.MAX_SELL_ENERGY)
    absolute_dir = os.path.abspath('')
    prod = np.load(absolute_dir + "/microGrid/env/data/BelgiumPV_prod_test.npy")[0:1 * 365 * 24]
    cons = np.load(absolute_dir + "/microGrid/env/data/example_nondeterminist_cons_test.npy")[0:1 * 365 * 24]
    env_valid = MG_two_storages_env(rng, consumption=cons, production=prod,
                                    pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                    length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                              max_ener_sell=EnvParam.MAX_SELL_ENERGY)

    # optimisation énergie
    env_ener = MG_two_storages_env(rng, consumption=cons, production=prod,
                                   pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                   length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                              max_ener_sell=EnvParam.MAX_SELL_ENERGY)
    #dict_env["energy"] = env_ener
    # ressenti client
    env_user = MG_two_storages_env(rng, consumption=cons, production=prod,
                                   pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                   length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                              max_ener_sell=EnvParam.MAX_SELL_ENERGY)
    #dict_env["user"] = env_user
    # profit réseau
    env_profit = MG_two_storages_env(rng, consumption=cons, production=prod,
                                     pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                     length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                              max_ener_sell=EnvParam.MAX_SELL_ENERGY)
    dict_env["profit"] = env_profit
    # préservation des stockages
    env_stockage = MG_two_storages_env(rng, consumption=cons, production=prod,
                                       pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                              max_ener_sell=EnvParam.MAX_SELL_ENERGY)
    #dict_env["stockage"] = env_stockage

    # --- Instantiate reward parameters ---
    price_h2 = 0.1  # 0.1euro/kWh of hydrogen
    price_elec_buy = 2.0  # 2euro/kWh
    cost_wast = 0.1  # arbitrary value
    reward_client = ClientReward()

    # --- comparative reward ---
    """dict_env["energy"].add_reward("Waste", lambda x: -x["waste_energy"] * cost_wast, 1.)
    dict_env["profit"].add_reward("Profit", lambda x: (x["sell_energy"] - x["buy_energy"]) * price_elec_buy, 1.)
    dict_env["user"].add_reward("Dissatisfaction", lambda x: reward_client.fn(x), 1.)"""

    # --- train reward ---
    env.add_reward("Flow_H2", lambda x: x["flow_H2"] * price_h2, 1.)
    env.add_reward("Buy_energy", lambda x: -x["buy_energy"] * price_elec_buy, 1.)

    # --- validation reward ---
    env_valid.add_reward("Flow_H2", lambda x: x["flow_H2"] * price_h2, 1.)
    env_valid.add_reward("Buy_energy", lambda x: -x["buy_energy"] * price_elec_buy, 1.)

    # --- init model ---
    print('MlpPolicy',
          "learning_rate=", Defaults.LEARNING_RATE,
          "buffer_size=", Defaults.REPLAY_MEMORY_SIZE,
          "batch_size=", Defaults.BATCH_SIZE,
          "gamma=", Defaults.DISCOUNT,
          "exploration_initial_eps=", Defaults.EPSILON_START,
          "exploration_final_eps=", Defaults.EPSILON_MIN,
          "exploration_fraction=", Defaults.EPSILON_DECAY, sep='\n')

    print(len(env.observation_space.sample()))

    model = DQN('MlpPolicy', env,
                learning_rate=Defaults.LEARNING_RATE,
                buffer_size=Defaults.REPLAY_MEMORY_SIZE,
                batch_size=Defaults.BATCH_SIZE,
                gamma=Defaults.DISCOUNT,
                exploration_initial_eps=Defaults.EPSILON_START,
                exploration_final_eps=Defaults.EPSILON_MIN,
                exploration_fraction=Defaults.EPSILON_DECAY,
                target_update_interval = Defaults.TARGET_UPDATE_INTERVAL,
                train_freq = 100,
                verbose=0)

    best = BestCallback(env_valid, dict_env, patience, filename, dirname)

    start = time.time()
    model.learn(Defaults.EPOCHS * Defaults.STEPS_PER_EPOCH,
                callback=best)  # callback=[verbose_callback, eps_callback, best_callback]
    res = time.time() - start
    print("time to train and valid:", int(res / 3600), "h", int((res % 3600) / 60), "min", res % 60, "s")
    plot_gene(best, dirname, filename, verbose=True)

def plot_gene(best, dirname, filename, verbose=False):

    data = best.get_data()
    bestScores, allScores = best.get_score()
    print(data.keys())

    actions = data["action"]
    consumption = data["consumption"]
    production = data["production"]
    rewards = data["rewards"]
    battery_level = data["soc"]
    # plot_op(data["action"], data["consumption"], data["production"], data["rewards"], data["battery"], "test.png")
    i = 0
    plot_op(actions[0 + i:100 + i], consumption[0 + i:100 + i], production[0 + i:100 + i], rewards[0 + i:100 + i],
            battery_level[0 + i:100 + i], "testplot_winter_.png")
    plt.show()
    i = 180 * 24
    plot_op(actions[0 + i:100 + i], consumption[0 + i:100 + i], production[0 + i:100 + i], rewards[0 + i:100 + i],
            battery_level[0 + i:100 + i], "testplot_summer_.png")
    plt.show()
    i = 360 * 24
    plot_op(actions[0 + i:100 + i], consumption[0 + i:100 + i], production[0 + i:100 + i], rewards[0 + i:100 + i],
            battery_level[0 + i:100 + i], "testplot_winter2_.png")
    if verbose:
        plt.show()

    key = "flow_H2"
    plt.plot(range(31 * 24), data[key][:31 * 24], label=key, color='b')
    key = "Buy_energy"
    plt.plot(range(31 * 24), data[key][:31 * 24], label=key, color='r')

    plt.legend()
    plt.xlabel("Number of hours")
    plt.ylabel("Score")
    plt.savefig(dirname + "/" + filename + "_plots.pdf")
    plt.show()

    h = sns.jointplot(x=[battery_level[i] for i in range(len(actions)) if actions[i] == 0],
                      y=[consumption[i] - production[i] for i in range(len(actions)) if actions[i] == 0],
                      kind="hist", marginal_ticks=True)
    # JointGrid has a convenience function
    h.set_axis_labels('charge battery', 'demand', fontsize=16)
    plt.savefig(dirname + "/" + filename + "_plots_action0.pdf")
    if verbose:
        plt.show()

    h = sns.jointplot(x=[battery_level[i] for i in range(len(actions)) if actions[i] == 1],
                      y=[consumption[i] - production[i] for i in range(len(actions)) if actions[i] == 1],
                      kind="hist", marginal_ticks=True)
    # JointGrid has a convenience function
    h.set_axis_labels('charge battery', 'demand', fontsize=16)
    plt.savefig(dirname + "/" + filename + "_plots_action1.pdf")
    if verbose:
        plt.show()

    h = sns.jointplot(x=[battery_level[i] for i in range(len(actions)) if actions[i] == 2],
                      y=[consumption[i] - production[i] for i in range(len(actions)) if actions[i] == 2],
                      kind="hist", marginal_ticks=True)
    # JointGrid has a convenience function
    h.set_axis_labels('charge battery', 'demand', fontsize=16)
    plt.savefig(dirname + "/" + filename + "_plots_action2.pdf")
    if verbose:
        plt.show()

    demande = [consumption[i] - production[i] for i in range(len(actions))]
    print("demande moyenne : ", np.mean(demande))
    print("demande std : ", np.std(demande))

    corr = pd.DataFrame.from_dict(bestScores)
    corr = corr.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True)
    # plt.savefig(dirname + "/" + filename + "_heatmap.pdf")
    if verbose:
        plt.show()

    corr = pd.DataFrame.from_dict(data)
    corr = corr.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True)
    plt.savefig(dirname + "/" + filename + "_heatmap.pdf")
    plt.show()
    print("reward", np.sum(data["rewards"]))

    scaler = preprocessing.MinMaxScaler()

    for k in bestScores.keys():
        try:
            plt.plot(range(len(bestScores[k])), scaler.fit_transform(np.array(bestScores[k]).reshape(-1,1)),
                     label="score " + str(k))
        except Exception as e:
            print("error",k)
            print(e)
            plt.plot(range(len(bestScores[k])), bestScores[k], label="score " + str(k))

    plt.legend()
    plt.xlabel("best step")
    plt.ylabel("normalized score")
    plt.savefig(dirname + "/" + filename + "_scoresbest.pdf")
    if verbose:
        plt.show()

    for k in bestScores.keys():
        try:
            plt.plot(range(len(allScores[k])), allScores[k],
                     label="score " + str(k))
        except Exception as e:
            print("error",k)
            print(e)
            plt.plot(range(len(allScores[k])), allScores[k], label="score " + str(k))

    plt.legend()
    plt.xlabel("number epoch")
    plt.ylabel("normalized score")
    plt.savefig(dirname + "/" + filename + "_scores.pdf")
    if verbose:
        plt.show()


    labels = 'discharge', 'none', 'charge'
    sizes = [len([1 for i in range(len(actions)) if actions[i] == 0]),
             len([1 for i in range(len(actions)) if actions[i] == 1]),
             len([1 for i in range(len(actions)) if actions[i] == 2])]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']

    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)

    plt.axis('equal')

    # plt.savefig('PieChart01.png')
    if verbose:
        plt.show()


if __name__ == "__main__":
    main()
