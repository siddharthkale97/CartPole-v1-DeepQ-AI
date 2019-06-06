import gym
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import rmsprop, Adam
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from statistics import mean
import h5py

LEARNING_RATE = 1e-3
MAX_MEMORY = 1000000
BATCH_SIZE = 20
GAMMA = 0.95
EXPLORATION_DECAY = 0.995
EXPLORATION_MIN = 0.01


class ScoreEvaluator:

    def __init__(self, max_len, average_of_last_runs, model = None):
        self.max_len = max_len
        self.score_table = deque(maxlen=self.max_len)
        self.model = model
        self.average_of_last_runs = average_of_last_runs

    def store_score(self, episode, step):
        self.score_table.append([episode, step])

    def plot_evaluation(self, title = "Training"):
        print(self.model.summary()) if self.model is not None else print("Model not defined!")
        avg_score = mean(self.score_table[1])
        x = []
        y = []
        for i in range(len(self.score_table)):
            x.append(self.score_table[i][0])
            y.append(self.score_table[i][1])

        average_range = self.average_of_last_runs if self.average_of_last_runs is not None else len(x)
        plt.plot(x, y, label="score per run")
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--",
                 label="last " + str(average_range) + " runs average")
        title = "CartPole-v1 " + str(title)
        plt.title(title)
        plt.xlabel("Runs")
        plt.ylabel("Score")
        plt.show()

class Network:

    def __init__(self, observation_space, action_space):

        self.action_space = action_space
        self.memory = deque(maxlen=MAX_MEMORY)
        self.exploration_rate = 1.0

        self.model = Sequential()
        self.model.add(Dense(32, input_shape=(observation_space,), activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.action_space, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def take_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(0, self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        else:
            minibatch = random.sample(self.memory, BATCH_SIZE)
            for state, action, reward, state_next, done in minibatch:
                Q = reward
                if not done:
                    Q = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
                Q_values = self.model.predict(state)
                Q_values[0][action] = Q
                self.model.fit(state, Q_values, verbose=0)
            self.exploration_rate *= EXPLORATION_DECAY
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def get_model(self):
        return self.model


class TrainSolver:

    def __init__(self, max_episodes):
        self.max_episodes = max_episodes
        self.score_table = deque(maxlen=400)
        self.average_of_last_runs = None
        self.model = None
        self.play_episodes = 100
        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        self.solver = Network(observation_space, action_space)

    def train(self):
        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n

        print("---------------------------------")
        print("Solver starts")
        print("---------------------------------")

        self.model = self.solver.get_model()
        episode = 0
        score_eval = ScoreEvaluator(400, 50, self.model)
        while episode < self.max_episodes:

            episode += 1
            state = env.reset()
            state = np.reshape(state, [1, observation_space])
            step = 0
            while True:

                step += 1
                # env.render()
                action = self.solver.take_action(state)
                state_next, reward, done, info = env.step(action)
                if not done:
                    reward = reward
                else:
                    reward = -reward
                state_next = np.reshape(state_next, [1, observation_space])
                self.solver.add_to_memory(state, action, reward, state_next, done)
                state = state_next

                if done:
                    print("Run: " + str(episode) + ", exploration: " + str(self.solver.exploration_rate) + ", score: " + str(step))
                    # self.score_table.append([episode, step])
                    score_eval.store_score(episode, step)
                    break
                self.solver.experience_replay()
        score_eval.plot_evaluation("Training")

    def return_trained_model(self):
        return self.model

    def play(self, play_episodes=100, load_model=False, model_weights_dir=None, trained_model=None):

        self.play_episodes = play_episodes
        if load_model is not False:
            if model_weights_dir is None:
                print("Can't load specified model")
            elif trained_model is None:
                print("Please pass a valid model as a parameter")
            else:
                model = trained_model
                model.load(model_weights_dir)
        else:
            model = self.model

        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        episode = 0
        score_eval = ScoreEvaluator(400, 100, model)
        while episode < self.play_episodes:

            episode += 1
            state = env.reset()
            state = np.reshape(state, [1, observation_space])
            step = 0
            while True:

                step += 1
                env.render()
                action = self.solver.take_action(state)
                state_next, reward, done, info = env.step(action)

                if not done:
                    reward = reward
                else:
                    reward = -reward
                state_next = np.reshape(state_next, [1, observation_space])
                self.solver.add_to_memory(state, action, reward, state_next, done)
                state = state_next

                if done:
                    print("Run: " + str(episode) + ", score: " + str(
                        step))
                    # self.score_table.append([episode, step])
                    score_eval.store_score(episode, step)
                    break
                self.solver.experience_replay()
        score_eval.plot_evaluation("100 Plays")

    def save_model(self):
        self.model.save('cartpole_model.h5')


if __name__ == "__main__":
    trainer = TrainSolver(150)
    trainer.train()
    trainer.play(100)
    trainer.save_model()
