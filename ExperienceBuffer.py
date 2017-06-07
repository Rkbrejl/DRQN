import numpy as np
import random


class ExperienceBuffer:

    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.id = 0

    def add(self, experience):
        exp = np.array(experience)
        print(len(exp))
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        print(np.array(self.buffer).shape)

    def sample(self, batch_size, trace_length):
        sampled_traces = []
        while i < batch_size:
            sampled_episode = random.sample(self.buffer, 1)
            for episode in sampled_episode:
                point = np.random.randint(0, len(episode) + 1 - trace_length)
                sampled_traces.append(episode[point:point + trace_length])
                i += 1
        sampled_traces = np.array(sampled_traces)
        return np.reshape(sampled_traces, [batch_size * trace_length, 5])

