from random import randint, random

import numpy as np

from Network import Network
from ExperienceBuffer import ExperienceBuffer
from ReplayMemory import ReplayMemory


class Agent:
    def __init__(self, memory_cap, batch_size, resolution, action_count, session,
                 lr, gamma, epsilon_min, epsilon_decay_steps, epsilon_max, trace_length, hidden_size):

        self.model = Network(session=session, action_count=action_count,
                             resolution=resolution, lr=lr, batch_size=batch_size,
                             trace_length=trace_length, hidden_size=hidden_size, scope='main')
        self.target_model = Network(session=session, action_count=action_count,
                                    resolution=resolution, lr=lr, batch_size=batch_size,
                                    trace_length=trace_length, hidden_size=hidden_size, scope='target')

        self.experience_buffer = ExperienceBuffer(buffer_size=memory_cap)

        self.memory = ReplayMemory(memory_cap=memory_cap, batch_size=batch_size,
                                   resolution=resolution, trace_length=trace_length)

        self.batch_size = batch_size

        self.resolution = resolution
        self.action_count = action_count
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_max = epsilon_max
        self.hidden_size = hidden_size
        self.trace_length = trace_length

        self.epsilon = epsilon_max
        self.training_steps = 0

        self.epsilon_decrease = (epsilon_max-epsilon_min)/epsilon_decay_steps

        self.min_buffer_size = batch_size*trace_length

        self.state_in = (np.zeros([1, self.hidden_size]), np.zeros([1, self.hidden_size]))

    def add_experience_to_buffer(self, episode_buffer):
        self.experience_buffer.add(episode_buffer)

    def add_transition(self, s1, a, r, s2, d):
        self.memory.add_transition(s1, a, r, s2, d)

    def learn_from_memory(self):

        if self.memory.size > self.min_buffer_size:
            state_in = (np.zeros([self.batch_size, self.hidden_size]), np.zeros([self.batch_size, self.hidden_size]))
            s1, a, r, s2, d = self.memory.get_transition()
            inputs = s1

            q = np.max(self.target_model.get_q(s2, state_in), axis=1)
            targets = r + self.gamma * (1 - d) * q

            self.model.learn(inputs, targets, state_in, a)

    def act(self, state, train=True):
        if train:
            self.epsilon = self.explore(self.epsilon)
            if random() < self.epsilon:
                a = self.random_action()
            else:
                a, self.state_in = self.model.get_best_action(state, self.state_in)
                a = a[0]
        else:
            a, self.state_in = self.model.get_best_action(state, self.state_in)
            a = a[0]
        return a

    def explore(self, epsilon):
        return max(self.epsilon_min, epsilon-self.epsilon_decrease)

    def random_action(self):
        return randint(0, self.action_count - 1)

    def reset_cell_state(self):
        self.state_in = (np.zeros([1, self.hidden_size]), np.zeros([1, self.hidden_size]))
