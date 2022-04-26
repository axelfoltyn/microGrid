import logging

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

import sys
print(os.path.abspath(''))
from os import path
sys.path.append( path.abspath(os.path.abspath('')) ) 
print(path.abspath(os.path.abspath('')) )

import time
from microGrid.env.final_env import MyEnv as MG_two_storages_env
from microGrid.callback.callback import BestCallback

from datetime import datetime
from microGrid.plot_MG_operation import plot_op
import tensorflow as tf
import pandas as pd
from stable_baselines3 import DQN


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
    BATCH_SIZE = 32
    FREEZE_INTERVAL = 1000
    DETERMINISTIC = False

def main():
    now = datetime.now()
    # dd_mm_YY-H-M-S
    dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")
    dirname = "result"
    filename = "best" + dt_string

    print("tensorflow work with:", tf.test.gpu_device_name())
    logging.basicConfig(level=logging.INFO)
    rng = np.random.RandomState()

    # --- Instantiate environment ---
    pred = 0
    equinox = 1
    dict_env = dict()

    env = MG_two_storages_env(rng, pred=pred, dist_equinox=equinox)
    absolute_dir = os.path.abspath('')
    prod = np.load(absolute_dir + "/microGrid/env/data/BelgiumPV_prod_test.npy")[0:1 * 365 * 24]
    cons = np.load(absolute_dir + "/microGrid/env/data/example_nondeterminist_cons_test.npy")[0:1 * 365 * 24]
    env_valid = MG_two_storages_env(rng, consumption=cons, production=prod, pred=pred, dist_equinox=equinox)

    # optimisation énergie
    env_ener = MG_two_storages_env(rng, consumption=cons, production=prod, pred=pred, dist_equinox=equinox)
    dict_env["energy"] = env_ener
    # ressenti client
    env_user = MG_two_storages_env(rng, consumption=cons, production=prod, pred=pred, dist_equinox=equinox)
    dict_env["user"] = env_user
    # profit réseau
    env_profit = MG_two_storages_env(rng, consumption=cons, production=prod, pred=pred, dist_equinox=equinox)
    dict_env["profit"] = env_profit
    # préservation des stockages
    env_stockage = MG_two_storages_env(rng, consumption=cons, production=prod, pred=pred, dist_equinox=equinox)
    dict_env["stockage"] = env_stockage

    # --- Instantiate reward parameters ---
    price_h2 = 0.1  # 0.1euro/kWh of hydrogen
    price_elec_buy = 2.0  # 2euro/kWh
    cost_wast = 0.1  # arbitrary value

    # --- train reward ---
    env.add_reward("flow_h2", lambda x: x["flow_H2"] * price_h2, 1.)
    env.add_reward("buy_energy", lambda x: -x["lack_energy"] * price_elec_buy, 1.)

    # --- validation reward ---
    env_valid.add_reward("flow_h2", lambda x: x["flow_H2"] * price_h2, 1.)
    env_valid.add_reward("buy_energy", lambda x: -x["lack_energy"] * price_elec_buy, 1.)

    # --- comparative reward ---
    dict_env["energy"].add_reward("waste", lambda x: -x["waste_energy"] * cost_wast, 1.)
    dict_env["profit"].add_reward("profit", lambda x: (x["waste_energy"] - x["lack_energy"]) * price_elec_buy, 1.)

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
    print(len(env.reset()))
    model = DQN('MlpPolicy', env,
                learning_rate=Defaults.LEARNING_RATE,
                buffer_size=Defaults.REPLAY_MEMORY_SIZE,
                batch_size=Defaults.BATCH_SIZE,
                gamma=Defaults.DISCOUNT,
                exploration_initial_eps=Defaults.EPSILON_START,
                exploration_final_eps=Defaults.EPSILON_MIN,
                exploration_fraction=Defaults.EPSILON_DECAY,
                verbose=0)


    best = BestCallback(env_valid, dict_env, 30, filename, dirname)
    start = time.time()
    model.learn(Defaults.EPOCHS * Defaults.STEPS_PER_EPOCH,
                callback=best)  # callback=[verbose_callback, eps_callback, best_callback]
    res = time.time() - start
    print("time to train and valid:", int(res / 3600), "h", int((res % 3600) / 60), "min", res % 60, "s")

    data = best.get_data()
    validationScores, trainScores = best.get_score()
    print(data.keys())

    actions = data["action"]
    consumption = data["consumption"]
    production = data["production"]
    rewards = data["rewards"]
    battery_level = data["battery"]
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
    plt.show()

    key = "flow_H2"
    plt.plot(range(31 * 24), data[key][:31 * 24], label=key, color='b')
    key = "buy_energy"
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
    plt.show()

    h = sns.jointplot(x=[battery_level[i] for i in range(len(actions)) if actions[i] == 1],
                      y=[consumption[i] - production[i] for i in range(len(actions)) if actions[i] == 1],
                      kind="hist", marginal_ticks=True)
    # JointGrid has a convenience function
    h.set_axis_labels('charge battery', 'demand', fontsize=16)
    plt.savefig(dirname + "/" + filename + "_plots_action1.pdf")
    plt.show()

    h = sns.jointplot(x=[battery_level[i] for i in range(len(actions)) if actions[i] == 2],
                      y=[consumption[i] - production[i] for i in range(len(actions)) if actions[i] == 2],
                      kind="hist", marginal_ticks=True)
    # JointGrid has a convenience function
    h.set_axis_labels('charge battery', 'demand', fontsize=16)
    plt.savefig(dirname + "/" + filename + "_plots_action2.pdf")
    plt.show()

    demande = [consumption[i] - production[i] for i in range(len(actions))]
    print("demande moyenne : ", np.mean(demande))
    print("demande std : ", np.std(demande))

    corr = pd.DataFrame.from_dict(validationScores)
    corr = corr.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True)
    # plt.savefig(dirname + "/" + filename + "_heatmap.pdf")
    plt.show()


    corr = pd.DataFrame.from_dict(data)
    corr = corr.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True)
    plt.savefig(dirname + "/" + filename + "_heatmap.pdf")
    plt.show()
    print("reward", np.sum(data["rewards"]))

    for k in validationScores.keys():
        plt.plot(range(len(validationScores[k])), validationScores[k], label="validation " + str(k))
    plt.plot(range(len(trainScores)), trainScores, label="train score")
    # plt.plot(x, np.repeat(testScores, nb_rep), label="TS", color='r')
    plt.legend()
    plt.xlabel("Number of cycle")
    plt.ylabel("Score")
    plt.savefig(dirname + "/" + filename + "_scores.pdf")
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
    plt.show()

if __name__ == "__main__":
    main()