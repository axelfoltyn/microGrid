"""2-Storage Microgrid launcher. See the docs for more details about this experiment.

"""

import sys
import logging
import time

import numpy as np
from joblib import hash, dump, load
import os
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from deer.default_parser import process_args
from microGrid.agent.final_agent import NeuralAgent
from microGrid.learning_algos.q_net_keras import MyQNetwork
from microGrid.env.final_env import MyEnv as MG_two_storages_env
import microGrid.experiment.base_controllers as bc
from datetime import datetime
from plot_MG_operation import plot_op
import tensorflow as tf
import pandas as pd

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 365*24-1
    EPOCHS = 200
    STEPS_PER_TEST = 365*24-1
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
    print("tensorflow work with:", tf.test.gpu_device_name())
    logging.basicConfig(level=logging.INFO)
    
    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)

    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()
    
    if(parameters.param1 is not None and parameters.param1!="1"):
        # We Reduce the size of the time series so that the number of days is divisible by 4*parameters.param1
        # That way, the number of days in each season is divisible by parameters.param1 and it is thus possible
        # to reduce the variety of the data within each season in the time series by a factor of parameters.param1
        parameters.steps_per_epoch=parameters.steps_per_epoch-(parameters.steps_per_epoch%(24*4*int(parameters.param1)))-1

    # --- Instantiate environment ---
    env = MG_two_storages_env(rng, reduce_qty_data=parameters.param1, length_history=parameters.param2)
    # --- Instantiate reward_function ---
    price_h2 = 0.1  # 0.1euro/kWh of hydrogen
    price_elec_buy = 2.0  # 2euro/kWh
    env.add_reward("flow_h2", lambda x: x["flow_H2"] * price_h2, 1.)
    env.add_reward("buy_energy", lambda x: -x["lack_energy"] * price_elec_buy, 1.)
    env.add_reward("flow_h2_bias", lambda x: x["flow_H2"] * price_h2 + 1, 0)

    # --- Instantiate environment test ---
    absolute_dir = os.path.dirname(os.path.abspath(__file__))
    prod = np.load(absolute_dir + "/env/data/BelgiumPV_prod_test.npy")[0:1 * 365 * 24]
    cons = np.load(absolute_dir + "/env/data/example_nondeterminist_cons_test.npy")[0:1*365*24]
    env_test = MG_two_storages_env(rng, reduce_qty_data=parameters.param1, length_history=parameters.param2,
                                   consumption=cons, production=prod)
    # --- Instantiate reward_function ---
    env_test.add_reward("flow_h2", lambda x: x["flow_H2"] * price_h2, 1.)
    env_test.add_reward("buy_energy", lambda x: -x["lack_energy"] * price_elec_buy, 1.)
    env_test.add_reward("flow_h2_bias", lambda x: x["flow_H2"] * price_h2 + 1, 0)

    # --- Instantiate environment test ---
    absolute_dir = os.path.dirname(os.path.abspath(__file__))
    prod = np.load(absolute_dir + "/env/data/BelgiumPV_prod_test.npy")[0:1 * 365 * 24]
    cons = np.load(absolute_dir + "/env/data/example_nondeterminist_cons_test.npy")[0:1 * 365 * 24]
    env_valid = MG_two_storages_env(rng, reduce_qty_data=parameters.param1, length_history=parameters.param2,
                                   consumption=cons, production=prod)
    # --- Instantiate reward_function ---
    env_valid.add_reward("flow_h2", lambda x: x["flow_H2"] * price_h2, 1.)
    env_valid.add_reward("buy_energy", lambda x: -x["lack_energy"] * price_elec_buy, 1.)
    env_valid.add_reward("flow_h2_bias", lambda x: x["flow_H2"] * price_h2 + 1, 0)

    # --- Instantiate qnetwork ---
    qnetwork = MyQNetwork(
        env,
        parameters.rms_decay,
        parameters.rms_epsilon,
        parameters.momentum,
        parameters.clip_norm,
        parameters.freeze_interval,
        parameters.batch_size,
        parameters.update_rule,
        rng)
    
    # --- Instantiate agent ---
    agent = NeuralAgent(
        env,
        qnetwork,
        parameters.replay_memory_size,
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
        parameters.batch_size,
        rng)
    
    # --- Create unique filename for FindBestController ---
    h = hash(vars(parameters), hash_name="sha1")
    fname = "MG2S_" + h
    print("The parameters hash is: {}".format(h))
    print("The parameters are: {}".format(parameters))

    # --- Bind controllers to the agent ---
    # Before every training epoch (periodicity=1), we want to print a summary of the agent's epsilon, discount and 
    # learning rate as well as the training epoch number.
    agent.attach("verbose", bc.VerboseController(
        evaluate_on='epoch',
        periodicity=1))

    # During training epochs, we want to train the agent after every [parameters.update_frequency] action it takes.
    # Plus, we also want to display after each training episode (!= than after every training) the average bellman
    # residual and the average of the V values obtained during the last episode, hence the two last arguments.
    agent.attach("train", bc.TrainerController(
        evaluate_on='action',
        periodicity=parameters.update_frequency,
        show_episode_avg_V_value=True,
        show_avg_Bellman_residual=True))

    # Every epoch end, one has the possibility to modify the learning rate using a LearningRateController. Here we
    # wish to update the learning rate after every training epoch (periodicity=1), according to the parameters given.
    """agent.attach("lr", bc.LearningRateController(
        initial_learning_rate=parameters.learning_rate,
        learning_rate_decay=parameters.learning_rate_decay,
        periodicity=1))

    # Same for the discount factor.
    agent.attach("df", bc.DiscountFactorController(
        initial_discount_factor=parameters.discount,
        discount_factor_growth=parameters.discount_inc,
        discount_factor_max=parameters.discount_max,
        periodicity=1))"""
    agent.setLearningRate(parameters.learning_rate)
    agent.setDiscountFactor(parameters.discount)

    # As for the discount factor and the learning rate, one can update periodically the parameter of the epsilon-greedy
    # policy implemented by the agent. This controllers has a bit more capabilities, as it allows one to choose more
    # precisely when to update epsilon: after every X action, episode or epoch. This parameter can also be reset every
    # episode or epoch (or never, hence the resetEvery='none').
    agent.attach("epsilon", bc.EpsilonController(
        initial_e=parameters.epsilon_start,
        e_decays=parameters.epsilon_decay,
        e_min=parameters.epsilon_min,
        evaluate_on='action',
        periodicity=1,
        reset_every='none'))

    # We wish to discover, among all versions of our neural network (i.e., after every training epoch), which one
    # seems to generalize the best, thus which one has the highest validation score. However we also want to keep
    # track of a "true generalization score", the "test score". Indeed, what if we overfit the validation score ?
    # To achieve these goals, one can use the FindBestController along two InterleavedTestEpochControllers, one for
    # each mode (validation and test). It is important that the validationID and testID are the same than the id
    # argument of the two InterleavedTestEpochControllers (implementing the validation mode and test mode
    # respectively). The FindBestController will dump on disk the validation and test scores for each and every
    # network, as well as the structure of the neural network having the best validation score. These dumps can then
    # used to plot the evolution of the validation and test scores (see below) or simply recover the resulting neural
    # network for your application.
    """agent.attach("best", bc.FindBestController(
        validationID=VALIDATION_MODE,
        testID=TEST_MODE,
        unique_fname=fname))"""

    # All previous controllers control the agent during the epochs it goes through. However, we want to interleave a
    # "validation epoch" between each training epoch (hence the periodicity=1). For each validation epoch, we want also
    # to  display the sum of all rewards obtained, hence the showScore=True. Finally, we never want this controller to
    # call the summarizePerformance method of MG_two_storage_env.
    agent.attach("validation", bc.InterleavedTestEpochController(
        id="validation",
        epoch_length=parameters.steps_per_epoch,
        periodicity=1,
        show_score=True,
        summarize_every=-1))

    # Besides inserting a validation epoch (required if one wants to find the best neural network over all training
    # epochs), we also wish to interleave a "test epoch" between each training epoch. For each test epoch, we also
    # want to display the sum of all rewards obtained, hence the showScore=True. Finally, we want to call the
    # summarizePerformance method of MG_two_storage_env every [parameters.period_btw_summary_perfs] *test* epochs.
    """agent.attach("test", bc.InterleavedTestEpochController(
        id="test",
        epoch_length=parameters.steps_per_test,
        periodicity=1,
        show_score=True,
        summarize_every=parameters.period_btw_summary_perfs))"""
    
    # --- Run the experiment ---
    try:
        os.mkdir("params")
    except Exception:
        pass
    dump(vars(parameters), "params/" + fname + ".jldump")
            
    #agent.run(parameters.epochs, parameters.steps_per_epoch)
    bestValidationScoreSoFar = -9999999
    validationScores = []
    trainScores = []
    testScores = []
    now = datetime.now()
    # dd_mm_YY-H-M-S
    dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")
    filename = "best" + dt_string

    step_before_test = 3
    step_before_test = min(step_before_test, parameters.epochs)
    first_turn = True
    eps = agent.getEpsilon()

    patience = 10
    cycle = 0
    epoch = 0
    try:
        start = time.time()
        #for epoch in range(max(1,int(parameters.epochs / step_before_test))):
        while "we don't lose our patience":
            cycle+=1
            epoch += 1
            # train part
            agent.setEpsilon(eps)
            agent.set_env(env, True)
            agent.setControllersActive("verbose", True)
            agent.setControllersActive("train", True)
            #agent.setControllersActive("test", False)
            agent.setControllersActive("validation", False)

            #agent.setControllersActive("df", True)
            #agent.setControllersActive("lr", True)
            agent.setControllersActive("epsilon", True)
            agent.run(step_before_test, parameters.steps_per_epoch, first_turn)
            first_turn = False
            eps = agent.getEpsilon()
            score, _ = agent.totalRewardOverLastTest()
            trainScores.append(score)

            # part validation
            # best action each time
            agent.setEpsilon(-1)

            agent.set_env(env_valid, False)
            agent.setControllersActive("verbose", False)
            agent.setControllersActive("train", False)
            #agent.setControllersActive("test", False)
            agent.setControllersActive("validation", True)

            #agent.setControllersActive("df", False)
            #agent.setControllersActive("lr", False)
            agent.setControllersActive("epsilon", False)
            agent.run(1, parameters.steps_per_epoch, first_turn)
            score, _ = agent.totalRewardOverLastTest()
            validationScores.append(score)
            """# part test
            agent.set_env(env_test, False)
            agent.setControllersActive("test", True)
            agent.setControllersActive("validation", False)
    
            agent.run(1, parameters.steps_per_epoch, first_turn)
            score, _ = agent.totalRewardOverLastTest()
            testScores.append(score)"""

            # part best
            if validationScores[-1] > bestValidationScoreSoFar:
                cycle = 0
                bestValidationScoreSoFar = validationScores[-1]
                print("new best", filename)
                agent.dumpNetwork(filename + "-score-" + str(validationScores[-1]), epoch)
            if cycle >= patience:
                break
    except KeyboardInterrupt:
        print('Hello user you have KeyboardInterrupt.')
    print("temps d'éxécution", time.time() - start)
    print("==========>", validationScores)
    env.end()
    env_test.end()
    # --- Show results ---
    """basename = "scores/" + fname
    scores = load(basename + "_scores.jldump")"""
    """plt.plot(range(1, len(scores['vs'])+1), scores['vs'], label="VS", color='b')
    plt.plot(range(1, len(scores['ts'])+1), scores['ts'], label="TS", color='r')"""




    data = env_valid.get_data()[-1]
    actions= data["action"]
    consumption = data["consumption"]
    production = data["production"]
    rewards = data["rewards"]
    battery_level= data["battery"]
    #plot_op(data["action"], data["consumption"], data["production"], data["rewards"], data["battery"], "test.png")
    i = 0
    plot_op(actions[0+i:100+i],consumption[0+i:100+i],production[0+i:100+i],rewards[0+i:100+i],battery_level[0+i:100+i],"testplot_winter_.png")
    plt.show()
    i=180*24
    plot_op(actions[0+i:100+i],consumption[0+i:100+i],production[0+i:100+i],rewards[0+i:100+i],battery_level[0+i:100+i],"testplot_summer_.png")
    plt.show()
    i=360*24
    plot_op(actions[0+i:100+i],consumption[0+i:100+i],production[0+i:100+i],rewards[0+i:100+i],battery_level[0+i:100+i],"testplot_winter2_.png")
    plt.show()
    print(data.keys())
    key = "flow_H2"
    print(key, len(data[key]))
    plt.plot(range(31*24), data[key][:31*24], label=key, color='b')
    key = "buy_energy"
    plt.plot(range(31*24), data[key][:31*24], label=key, color='r')
    print(key, len(data[key]))
    plt.legend()
    plt.xlabel("Number of hours")
    plt.ylabel("Score")
    plt.savefig(filename + "_plots.pdf")
    plt.show()

    h = sns.jointplot(x=[battery_level[i] for i in range(len(actions)) if actions[i] == 0],
                      y=[consumption[i] - production[i] for i in range(len(actions)) if actions[i] == 0],
                      kind="hist")
    # JointGrid has a convenience function
    h.set_axis_labels('charge battery', 'demand', fontsize=16)
    plt.savefig(filename + "_plots_action0.pdf")
    plt.show()

    h = sns.jointplot(x=[battery_level[i] for i in range(len(actions)) if actions[i] == 1],
                      y=[consumption[i] - production[i] for i in range(len(actions)) if actions[i] == 1],
                      kind="hist")
    # JointGrid has a convenience function
    h.set_axis_labels('charge battery', 'demand', fontsize=16)
    plt.savefig(filename + "_plots_action1.pdf")
    plt.show()

    h = sns.jointplot(x=[battery_level[i] for i in range(len(actions)) if actions[i] == 2],
                      y=[consumption[i] - production[i] for i in range(len(actions)) if actions[i] == 2],
                      kind="hist")
    # JointGrid has a convenience function
    h.set_axis_labels('charge battery', 'demand', fontsize=16)
    plt.savefig(filename + "_plots_action2.pdf")
    plt.show()

    plt.plot(range(len(validationScores)), validationScores, label="validation score", color='b')
    plt.plot(range(len(trainScores)), trainScores, label="train score", color='r')
    #plt.plot(x, np.repeat(testScores, nb_rep), label="TS", color='r')
    plt.legend()
    plt.xlabel("Number of cycle")
    plt.ylabel("Score")
    plt.savefig(filename + "_scores.pdf")
    plt.show()
    demande = [consumption[i] - production[i] for i in range(len(actions))]
    print("demande moyenne : ", np.mean(demande))
    print("demande std : ", np.std(demande))

    print("data",data)
    corr = pd.DataFrame.from_dict(data)
    print("pd", corr)
    corr = corr.corr()
    print("corr", corr)
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot = True)
    plt.show()

if __name__ == "__main__":
    main()