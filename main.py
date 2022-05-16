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
from microGrid.reward.reward import ClientReward, BlackoutReward

from datetime import datetime
import shutil

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
    STEPS_PER_EPISODE = 365 * 24 - 1
    EPISODE = 200
    STEPS_PER_TEST = 365 * 24 - 1

    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 1

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

class EnvParam:
    MAX_BUY_ENERGY = None
    MAX_SELL_ENERGY = 0
    PREDICTION = False
    EQUINOX = True
    LENGTH_HISTORY = 12

def init_env():
    rng = np.random.RandomState()
    env_res = dict()
    env_valid_res = dict()
    absolute_dir = os.path.abspath('')
    prod = np.load(absolute_dir + "/microGrid/env/data/BelgiumPV_prod_test.npy")[0:1 * 365 * 24]
    cons = np.load(absolute_dir + "/microGrid/env/data/example_nondeterminist_cons_test.npy")[0:1 * 365 * 24]

    # --- Instantiate reward parameters ---
    price_h2 = 0.1  # 0.1euro/kWh of hydrogen
    price_elec_buy = 2.0  # 2euro/kWh
    cost_wast = 0.1  # arbitrary value
    
    # Fixme : il faut reset pour l'automatisation
    reward_client = ClientReward()
    reward_client_valid = ClientReward()
    reward_blackout = BlackoutReward()
    reward_valid_blackout = BlackoutReward()

    key = "default"
    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                              length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                              max_ener_sell=EnvParam.MAX_SELL_ENERGY)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                    pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                    length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                                    max_ener_sell=EnvParam.MAX_SELL_ENERGY)

    env_res[key].add_reward("Flow_H2", lambda x: x["flow_H2"] * price_h2, 1.)
    env_res[key].add_reward("Buy_energy", lambda x: -x["buy_energy"] * price_elec_buy, 1.)
    env_valid_res[key].add_reward("Flow_H2", lambda x: x["flow_H2"] * price_h2, 1.)
    env_valid_res[key].add_reward("Buy_energy", lambda x: -x["buy_energy"] * price_elec_buy, 1.)

    # optimisation énergie
    key = "- perte_d’energie * cout_perte"
    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                              length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                              max_ener_sell=EnvParam.MAX_SELL_ENERGY)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                    pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                    length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                                    max_ener_sell=EnvParam.MAX_SELL_ENERGY)
    env_res[key].add_reward("Waste", lambda x: -x["waste_energy"] * cost_wast, 1.)
    env_valid_res[key].add_reward("Waste", lambda x: -x["waste_energy"] * cost_wast, 1.)

    # ressenti client
    key = "f_insatisfait(temps_sans_energie)"
    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                                       max_ener_sell=EnvParam.MAX_SELL_ENERGY)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY,
                                             max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                                             max_ener_sell=EnvParam.MAX_SELL_ENERGY)
    env_res[key].add_reward("Dissatisfaction", lambda x: reward_client.fn(x), 1.)
    env_valid_res[key].add_reward("Dissatisfaction", lambda x: reward_client_valid.fn(x), 1.)


    # profit réseau
    key = "vente − achat"
    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                                       max_ener_sell=EnvParam.MAX_SELL_ENERGY)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY,
                                             max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                                             max_ener_sell=EnvParam.MAX_SELL_ENERGY)
    env_res[key].add_reward("Profit", lambda x: (x["sell_energy"] - x["buy_energy"]) * price_elec_buy, 1.)

    env_valid_res[key].add_reward("Profit", lambda x: (x["sell_energy"] - x["buy_energy"]) * price_elec_buy, 1.)

    # profit réseau
    key = "- prix_achat * achat"
    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                                       max_ener_sell=EnvParam.MAX_SELL_ENERGY)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY,
                                             max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                                             max_ener_sell=EnvParam.MAX_SELL_ENERGY)
    env_res[key].add_reward("Profit", lambda x: (-x["buy_energy"]) * price_elec_buy, 1.)
    env_valid_res[key].add_reward("Profit", lambda x: (-x["buy_energy"]) * price_elec_buy, 1.)

    # préservation des stockages
    key = "stockage"

    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                                       max_ener_sell=EnvParam.MAX_SELL_ENERGY)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY,
                                             max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                                             max_ener_sell=EnvParam.MAX_SELL_ENERGY)

    env_res[key].add_reward("Dissatisfaction", lambda x: reward_client.fn(x), 1.)
    env_valid_res[key].add_reward("Dissatisfaction", lambda x: reward_client.fn(x), 1.)

    key = "- cout_coupure * coupure"

    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                                       max_ener_sell=EnvParam.MAX_SELL_ENERGY)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY,
                                             max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                                             max_ener_sell=EnvParam.MAX_SELL_ENERGY)

    env_res[key].add_reward("Dissatisfaction", lambda x: reward_blackout.fn(x), 1.)
    env_valid_res[key].add_reward("Dissatisfaction", lambda x: reward_valid_blackout.fn(x), 1.)

    key = "flux_batterie h2"

    env_res[key] = MG_two_storages_env(rng, pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                       length_history=EnvParam.LENGTH_HISTORY, max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                                       max_ener_sell=EnvParam.MAX_SELL_ENERGY)

    env_valid_res[key] = MG_two_storages_env(rng, consumption=cons, production=prod,
                                             pred=EnvParam.PREDICTION, dist_equinox=EnvParam.EQUINOX,
                                             length_history=EnvParam.LENGTH_HISTORY,
                                             max_ener_buy=EnvParam.MAX_BUY_ENERGY,
                                             max_ener_sell=EnvParam.MAX_SELL_ENERGY)

    env_res[key].add_reward("Dissatisfaction", lambda x: x["flow_H2"], 1.)
    env_valid_res[key].add_reward("Dissatisfaction", lambda x: x["flow_H2"], 1.)

    return env_res, env_valid_res


def main():
    patience = 5
    dirname = "result"

    # --- Instantiate environments ---
    env_res, env_valid_res = init_env()

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

    start = time.time()
    for key in env_res.keys():
        env = env_res[key]
        env_valid = env_valid_res[key]
        dict_env = {k:val for (k,val) in env_res.items() if k != key}
        now = datetime.now()
        # dd_mm_YY-H-M-S
        dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")
        filename = "best" + dt_string + "_env_" + key
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
             verbose=False, param=str(key))
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

    f = open(dirname+ "/" + filename + "/" + filename + "hyperparam.txt", "a")
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
    try:
        start = time.time()
        model.learn(Defaults.EPISODE * Defaults.STEPS_PER_EPISODE,
                    callback=best)  # callback=[verbose_callback, eps_callback, best_callback]
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
    if len(bestScores[list(bestScores.keys())[0]]) < 5:
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

    """key = "flow_H2"
    plt.plot(range(31 * 24), data[key][:31 * 24], label=key, color='b')
    key = "Buy_energy"
    plt.plot(range(31 * 24), data[key][:31 * 24], label=key, color='r')

    plt.legend()
    plt.xlabel("Number of hours")
    plt.ylabel("Score")
    plt.savefig(dirname + "/" + filename + "/" + filename + "_plots.png")
    if verbose:
        plt.show()
    plt.clf()"""

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

    corr = pd.DataFrame.from_dict(bestScores)
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
    sizes = [len([1 for i in range(len(actions)) if actions[i] == 0]),
             len([1 for i in range(len(actions)) if actions[i] == 1]),
             len([1 for i in range(len(actions)) if actions[i] == 2])]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']

    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)

    plt.axis('equal')
    plt.title("camembert des proportion des actions choisies")

    plt.savefig(dirname + "/" + filename + "/" + filename + '_PieChart.png', bbox_inches = "tight")
    if verbose:
        plt.show()
    plt.clf()
    plt.close("all")

    with open(dirname + "/" + filename + "/" + filename + "_data.csv", 'w') as f:
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



if __name__ == "__main__":
    main()
