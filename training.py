import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from game_state import GameState
from memory import ReplayMemory
from model import DQNetwork

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.0001
LR = 1e-4

model = None
counterpart = None
memory = None
optimizer = None
steps_done = 0

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

def select_action(state):
    # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        #print("Using model to sample step")
        with torch.no_grad():
            return model(torch.tensor(state.vector.flatten(), dtype=torch.float64)).max(-1).indices.view(1, 1)
    else:
        #print("Using random option")
        return torch.tensor([[np.random.randint(8)]], device=device, dtype=torch.long)
    
def determine_batch_loss(rewards, estimated_rewards, actions):
    index = torch.arange(0, BATCH_SIZE)
    # Combination of row index and action (output index) to consider
    index2d = torch.stack([index, actions], axis=1)
    return nn.MSELoss(estimated_rewards[index2d], rewards)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    recordings = memory.sample(BATCH_SIZE)

    state_vectors = torch.tensor([torch.tensor(recording.game_state.vector.flatten(), dtype=torch.float64) for recording in recordings])
    actions = torch.tensor([recording.action for recording in recordings])
    rewards = torch.tensor([recording.reward for recording in recordings])

    estimated_rewards = model(state_vectors)
    loss = determine_batch_loss(rewards, estimated_rewards, actions)
    loss.backward()
    optimizer.step()

def play_game(start=True, debug=False):
    state = GameState()

    state_history = []
    actions_taken = []
    
    winning_reward = 1
    losing_reward = -1
    rewards = []

    # Make first move if protagonist (red; player 1) starts
    if start:
        action = select_action(state)
        state_history.append(state)
        actions_taken.append(action)
        state = GameState(previous_state=state, index=action, red=True)
        if debug:
            print(state.vector.transpose()[::-1, :])


    while True:
        model_result = counterpart(torch.tensor(state.vector.flatten(), dtype=torch.float64))
        counterpart_action = model_result.max(-1).indices.view(1, 1)
        state_history.append(state)
        actions_taken.append(counterpart_action)
        state = GameState(previous_state=state, index=counterpart_action, red=False)
        if debug:
            print(state.vector.transpose()[::-1, :])

        winner = state.determine_winner()
        if winner is not None:
            print("Winner: " + str(winner))
            first_player_won = int(start) if winner == 1 else int(not start)

            rewards = [winning_reward if (i+1) % 2 == first_player_won else losing_reward for i in range(len(actions_taken))]
            break

        action = select_action(state)
        state_history.append(state)
        actions_taken.append(action)
        state = GameState(previous_state=state, index=action, red=True)
        if debug:
            print(state.vector.transpose()[::-1, :])

        winner = state.determine_winner()
        if winner is not None:
            print("Winner: " + str(winner))
            first_player_won = int(start) if winner == 1 else int(not start)
            rewards = [winning_reward if (i+1) % 2 == first_player_won else losing_reward for i in range(len(actions_taken))]
            break

    # Add experience to memory
    for i in range(len(actions_taken)):
        memory.push(state_history[i], actions_taken[i], rewards[i])

def adapt_counterpart():
    counterpart_state_dict = counterpart.state_dict()
    protagonist_state_dict = model.state_dict()
    for key in protagonist_state_dict:
        counterpart_state_dict[key] = protagonist_state_dict[key]*TAU + counterpart_state_dict[key]*(1-TAU)
    counterpart.load_state_dict(counterpart_state_dict)

def train():
    global model
    model = DQNetwork()

    global counterpart
    counterpart = DQNetwork()

    global memory
    memory = ReplayMemory(capacity=100)

    global optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, amsgrad=True)

    for i in range(100):
        print("Iteration " + str(i))
        for j in range(10):
            play_game(debug=False)
            #memory.print()
            optimize_model()

            adapt_counterpart()
