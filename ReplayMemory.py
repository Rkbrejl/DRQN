from random import sample
import numpy as np


class ReplayMemory:
    def __init__(self, memory_cap, batch_size, resolution, trace_length):

        state_shape = (memory_cap, resolution[0], resolution[1], resolution[2])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(memory_cap, dtype=np.int32)
        self.r = np.zeros(memory_cap, dtype=np.float32)
        self.d = np.zeros(memory_cap, dtype=np.float32)

        self.memory_cap = memory_cap
        self.batch_size = batch_size
        self.trace_length = trace_length
        self.index = 0
        self.size = 0

    def add_transition(self, s1, a, r, s2, d):

        self.s1[self.index, :, :, :] = s1
        self.a[self.index] = a
        self.r[self.index] = r
        self.s2[self.index, :, :, :] = s2
        self.d[self.index] = d

        self.index = (self.index+1) % self.memory_cap
        self.size = min(self.size + 1, self.memory_cap)

    def get_transition(self):
        indexes = []
        for _ in range(self.batch_size):
            accepted = False
            while not accepted:
                point = np.random.randint(0, self.size - self.trace_length)
                accepted = True
                for i in range(self.trace_length-1):
                    if self.d[point+i] > 0:
                        accepted = False
                        break
                if accepted:
                    for i in range(self.trace_length):
                        indexes.append(point+i)

        return self.s1[indexes], self.a[indexes], self.r[indexes], self.s2[indexes], self.d[indexes]
