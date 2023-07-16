import random
import torch
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0 

    def rest(self):
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done =  zip(*batch)
        return state, action, reward, next_state, done

    def get_all(self):
        state, action, reward, next_state, done = zip(*self.buffer)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class RecommondReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity 
        self.buffer = [] 
        self.position = 0

    def rest(self):
        self.buffer = []
        self.position = 0

    def push(self, user, memory, memory_rate, action, reward, next_memory, next_memory_rate, done, mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (user, memory, memory_rate, action, reward, next_memory, next_memory_rate, done, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, update_size):
        batch = random.sample(self.buffer, update_size)
        user, memory, memory_rate, action, reward, next_memory, next_memory_rate, done, mask = zip(*batch)
        return user, memory, memory_rate, action, reward, next_memory, next_memory_rate, done, mask

    def get_all(self):
        user, memory, memory_rate, action, reward, next_memory, next_memory_rate, done, mask = zip(*self.buffer)
        return  user, memory, memory_rate, action, reward, next_memory, next_memory_rate, done, mask

    def __len__(self):
        return len(self.buffer)

class RealCounterFactualReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def rest(self):
        self.buffer = []
        self.position = 0

    def push(self, user, memory, memory_rate, action, reward, next_memory, next_memory_rate, done, mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (user, memory, memory_rate, action, reward, next_memory, next_memory_rate, done, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, update_size):
        batch = random.sample(self.buffer, update_size)
        user, memory, memory_rate, action, reward, next_memory, next_memory_rate, done, mask = zip(*batch)
        return user, memory, memory_rate, action, reward, next_memory, next_memory_rate, done, mask

    def get_all(self):
        user, memory, memory_rate, action, reward, next_memory, next_memory_rate, done, mask = zip(*self.buffer)
        return user, memory, memory_rate, action, reward, next_memory, next_memory_rate, done, mask

    def __len__(self):
        return len(self.buffer)
