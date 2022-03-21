from microGrid.env.env3 import MicroGrid2
import numpy as np

if __name__ == '__main__':
    env = MicroGrid2(np.random.RandomState())
    env_test = MicroGrid2(np.random.RandomState())
    # --- Instantiate reward_function ---
    price_h2 = 0.1  # 0.1euro/kWh of hydrogen
    price_elec_buy = 2.0  # 2euro/kWh
    env.add_reward("flow_H2", lambda x: x["flow_H2"] * price_h2, 1.)
    env.add_reward("buy_energy", lambda x: -x["lack_energy"] * price_elec_buy, 1.)
    env.add_reward("flow_H2_bias", lambda x: x["flow_H2"] * price_h2 + 1, 1.)

    env.add_reward("flow_H2", lambda x: x["flow_H2"] * price_h2, 1.)
    env_test.add_reward("buy_energy", lambda x: -x["lack_energy"] * price_elec_buy, 1.)
    env_test.add_reward("flow_H2_bias", lambda x: x["flow_H2"] * price_h2 + 1, 2.)

    nb_ep_before_test = 2

    episodes = 10
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            # env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward

        print('Episode:{} Score:{}'.format(episode, score))

        if episode % nb_ep_before_test == 0:
            print("Start test")
            state = env_test.reset()
            done = False
            score_test = 0

            while not done:
                # env.render()
                action = env_test.action_space.sample()
                n_state, reward, done, info = env_test.step(action)
                score_test += reward
            print('result test : Episode:{} Score:{}'.format(episode, score_test))
