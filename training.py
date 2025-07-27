import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from copy import deepcopy
from game_state import GameState
from memory import ReplayMemory
from model import DQNetwork

BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.0002
LR = 5e-4

model = None
counterpart = None
memory = None
optimizer = None
steps_done = 0

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

def select_action(state, debug=False):
    # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        #print("Using model to sample step")
        with torch.no_grad():
            output = model(torch.tensor(state.vector.reshape(1, 1, 8, 8), dtype=torch.float64))
            if debug:
                print("Predicted rewards:")
                print(output)
            return output.max(-1).indices.view(1, 1)
    else:
        #print("Using random option")
        return torch.tensor([[np.random.randint(8)]], device=device, dtype=torch.long)
    
def determine_batch_loss(rewards, estimated_rewards, actions):
    index = torch.arange(0, BATCH_SIZE)
    # Combination of row index and action (output index) to consider
    index2d = torch.stack([index, actions], axis=0)
    loss_fn = nn.MSELoss()
    return loss_fn(estimated_rewards[tuple(index2d)], rewards.double())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    recordings = memory.sample(BATCH_SIZE)

    state_vectors = torch.tensor([recording.game_state.vector.flatten() for recording in recordings], dtype=torch.float64)
    actions = torch.tensor([recording.action for recording in recordings])
    rewards = torch.tensor([recording.reward for recording in recordings])

    estimated_rewards = model(state_vectors.reshape(len(recordings), 1, 8, 8))
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

    # Last move rewards are multiplied by an ascending factor
    max_positive_reward = 3

    # Make first move if protagonist (red; player 1) starts
    if start:
        action = select_action(state, debug=debug)
        state_history.append(deepcopy(state))
        actions_taken.append(action)
        state = GameState(previous_state=state, index=action, red=True)
        if debug:
            print(state.vector.transpose()[::-1, :])


    while True:
        # Flip entries to take view of counterpart
        model_result = counterpart(-1 * torch.tensor(state.vector.reshape(1, 1, 8, 8), dtype=torch.float64))
        counterpart_action = model_result.max(-1).indices.view(1, 1)
        # Flip entries to take view of counterpart
        state_history.append(deepcopy(state.invert()))
        actions_taken.append(counterpart_action)
        state = GameState(previous_state=state, index=counterpart_action, red=False)
        if debug:
            print(state.vector.transpose()[::-1, :])

        winner = state.determine_winner()
        if winner is not None:
            print("Winner: " + str(winner))
            first_player_won = int(start) if winner == 1 else int(not start)
            # Last few plays are reflected stronger
            rewards = [max(1, (i-len(actions_taken)+max_positive_reward+1))*(winning_reward if (i+1) % 2 == first_player_won else losing_reward) for i in range(len(actions_taken))]
            break

        action = select_action(state, debug=debug)
        state_history.append(deepcopy(state))
        actions_taken.append(action)
        state = GameState(previous_state=state, index=action, red=True)
        if debug:
            print(state.vector.transpose()[::-1, :])

        winner = state.determine_winner()
        if winner is not None:
            print("Winner: " + str(winner))
            first_player_won = int(start) if winner == 1 else int(not start)
            # Last few plays are reflected stronger
            rewards = [max(1, (i-len(actions_taken)+6))*(winning_reward if (i+1) % 2 == first_player_won else losing_reward) for i in range(len(actions_taken))]
            break

    # Add experience to memory
    # If last reward was negative, so a player kicked himself out, penalize with -1e5 and ignore path to this ending
    if rewards[-1] == -1 * max_positive_reward:
        memory.push(state_history[-1], actions_taken[-1], -1e5)
    else:
        for i in range(len(actions_taken)):
            if debug:
                print("New Memory")
                print(state_history[i].vector.transpose()[::-1, :])
                print(actions_taken[i])
                print(rewards[i])
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
    memory = ReplayMemory(capacity=10000)

    global optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, amsgrad=True)

    for i in range(500):
        print("Iteration " + str(i))
        for j in range(10):
            start = np.random.binomial(size=1, n=1, p= 0.5)[0]
            play_game(start=start, debug=False)

            optimize_model()
            adapt_counterpart()

    print("Current state:")
    play_game(debug=True)