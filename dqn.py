import gym
import numpy as np
import pandas as pd

from keras.layers import Dense
from scipy.linalg import hankel
from keras.models import Sequential
from matplotlib import pyplot as plt
from replay_buffer import ReplayBuffer
from numpy.polynomial.polynomial import polyval

"""
        DQN agent implementation that uses monte carlo method to learn to play cart pole game.        
"""


class DQNAgent:
    def __init__(self, env_name='CartPole-v0', max_iter=100, gamma=0.9, epsilon=0.9,
                 epsilon_decay=0.05, buffer_size=2000, batch_size=50, stop_after_n_wins=3):
        self.model = None
        self.init_model()
        self.decay = epsilon_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.env_name = env_name
        self.current_state = None
        self.batch_size = batch_size
        self.env = gym.make(env_name)
        self.stop_after_n_wins = stop_after_n_wins
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.legal_actions = list(range(self.env.action_space.n))
        self.max_episode_steps = self.env.spec.max_episode_steps

    # plot rewards agent got from each episode
    @staticmethod
    def plot_rewards(rewards):
        plt.plot(rewards, label='sum of rewards')
        plt.legend()
        plt.xlabel('episodes')
        plt.show()

    # extract features and labels from dataframe to feed to dnn
    @staticmethod
    def extract_features_labels(data):
        ys = data['reward'].to_numpy()
        xs = data.drop(columns=['reward']).to_numpy()
        return xs, ys

    # create model architecture
    def init_model(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(5,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        self.model = model

    # discount the rewards
    def get_discounted_reward(self, rewards):
        return polyval(self.gamma, hankel(rewards))

    # decrease epsilon after every episode so agent's moves become more greedy over time
    def update_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon * self.decay, 0)

    # get random action with epsilon probability, otherwise pick action with most predicted reward
    def get_epsilon_greedy_action(self):
        if self.epsilon > np.random.uniform(0, 1):
            return np.random.choice(self.legal_actions)

        else:
            predicted_rewards = []
            for action in self.legal_actions:
                state_action = np.append(self.current_state, np.array(action))
                predicted_rewards.append(self.model.predict(state_action.reshape((1, 5)))[0][0])

            return np.argmax(predicted_rewards)

    # play and train dnn
    def play_and_train(self):
        reward_sums = []
        hm_times_won = 0
        for episode in range(self.max_iter):
            data_df = pd.DataFrame(columns=['state 1', 'state 2', 'state 3', 'state 4', 'action', 'reward'])
            self.current_state = self.env.reset()

            for iteration_step in range(self.max_episode_steps - 1):
                action = self.get_epsilon_greedy_action()
                next_state, reward, done, info = self.env.step(action)
                data_df.loc[iteration_step] = ([*self.current_state, action, reward])
                self.current_state = next_state

                if done:
                    break

            if iteration_step == self.max_episode_steps - 2:
                hm_times_won += 1
                if hm_times_won == self.stop_after_n_wins:
                    print(f'ended training after winning the game {hm_times_won} times')
                    self.plot_rewards(reward_sums)
                    return

            if episode % 10 == 0:
                print(f'current episode: {episode}')

            reward_sums.append(data_df['reward'].sum())

            discounted_rewards = self.get_discounted_reward(data_df['reward'].to_numpy())
            data_df['reward'] = discounted_rewards
            self.replay_buffer.add(data_df)

            xs, ys = self.extract_features_labels(data_df)
            self.model.fit(xs, ys, epochs=3, verbose=0)

            historic_data_df = self.replay_buffer.sample(self.batch_size)
            xs, ys = self.extract_features_labels(historic_data_df)
            self.model.fit(xs, ys, epochs=3, verbose=0)

            self.update_epsilon()

        self.plot_rewards(reward_sums)

    # demonstration play
    def play(self):
        self.epsilon = 0
        done = False
        total_rewards = 0
        self.current_state = self.env.reset()
        while not done:
            action = self.get_epsilon_greedy_action()
            self.env.render()
            self.current_state, reward, done, _ = self.env.step(action)
            total_rewards += reward
        print(f'total rewards of episode {total_rewards}')


dqn = DQNAgent(max_iter=1000, epsilon=0.95, gamma=0.99, buffer_size=1000, batch_size=32, epsilon_decay=0.003)
dqn.play_and_train()
dqn.play()
