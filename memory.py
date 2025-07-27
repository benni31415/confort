from collections import deque
import random

from recording import Recording


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Recording(*args))

    def sample(self, batch_size, last_n=0):
        return list(self.memory)[-last_n:] + random.sample(self.memory, batch_size-last_n)

    def __len__(self):
        return len(self.memory)
    
    def print(self):
        print(self.memory)