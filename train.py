import argparse
import os
import shutil
from random import random, sample
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from src.deep_q_network import DeepQNetwork
from flappy import Flappyplayer
from util import pre_processing


def get_args():
    args = argparse.ArgumentParser("Q Network to play Forked FlapPyBird")
    args.add_argument("--image_size", type=int, default=84)
    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--optimiser", type=str, choices=["adam", "sgd"], default="adam")
    args.add_argument("--lr", type=float, default=1e-6)
    args.add_argument("--gamma", type=float, default=0.99)
    args.add_argument("--initial_epsilon", type=float, default=0.1)
    args.add_argument("--final_epsilon", type=float, default=1e-4)
    args.add_argument("--iters", type=int, default=2000000)
    args.add_argument("--replay_mem", type=int, default=50000)
    args.add_argument("--log_path", type=str, default="tensorboard")
    args.add_argument("--saved_path", type=str, default="trained_models")
    parse = args.parse_args()
    return parse


def training(arguments):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    model = DeepQNetwork()
    if os.path.isdir(arguments.log_path):
        shutil.rmtree(arguments.log_path)
    os.makedirs(arguments.log_path)
    writer = SummaryWriter(arguments.log_path)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    gameState = Flappyplayer()
    image, reward, terminal = gameState.next_frame(0)
    image = pre_processing(image[:gameState.SCREENW, :int(gameState.base_y)], arguments.image_size,
                           arguments.image_size)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
    replay_mem = []
    iter = 0
    while iter < arguments.iters:
        prediction = model(state)[0]
        # Exploration or exploitation
        epsilon = arguments.final_epsilon + (
                (arguments.iters - iter) * (arguments.initial_epsilon - arguments.final_epsilon) / arguments.iters)
        u = random()
        random_action = u <= epsilon
        if random_action:
            print("Perform a random action")
            action = 1
            print(action)
        else:
            action = 0

        nextImage, reward, terminal = gameState.next_frame(action)
        nextImage = pre_processing(nextImage[:gameState.SCREENW, :int(gameState.base_y)], arguments.image_size,
                                   arguments.image_size)
        nextImage = torch.from_numpy(nextImage)
        if torch.cuda.is_available():
            nextImage = nextImage.cuda()
        nextState = torch.cat((state[0, 1:, :, :], nextImage))[None, :, :, :]
        replay_mem.append([state, action, reward, nextState, terminal])
        if len(replay_mem) > arguments.replay_mem:
            del replay_mem[0]

        batch = sample(replay_mem, min(len(replay_mem), arguments.batch_size))
        stateBatch, actionBatch, rewardBatch, nextStateBatch, terminalBatch = zip(*batch)

        stateBatch = torch.cat(tuple(state for state in stateBatch))
        actionBatch = torch.from_numpy(np.array([[1,0] if action == 0 else [0,1] for action in actionBatch], dtype=np.float32))
        rewardBatch = torch.from_numpy(np.array(rewardBatch, dtype=np.float32)[:, None])
        nextStateBatch = torch.cat(tuple(state for state in nextStateBatch))

        if torch.cuda.is_available():
            stateBatch = stateBatch.cuda()
            actionBatch = actionBatch.cuda()
            rewardBatch = rewardBatch.cuda()
            nextStateBatch = nextStateBatch.cuda()
        currentPredBatch = model(stateBatch)
        nextPredBatch = model(nextStateBatch)

        yBatch = torch.cat(tuple(reward if terminal else reward + arguments.gamma * torch.max(prediction) for reward, terminal, prediction in zip(rewardBatch, terminalBatch, nextPredBatch)))
        qValue = torch.sum(currentPredBatch*actionBatch, dim=1)
        optimiser.zero_grad()
        loss = criterion(qValue, yBatch)
        loss.backward()
        optimiser.step()

        state = nextState
        iter +=1
        print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon: {}, Reward: {}, Q-Value: {}".format(iter+1,arguments.iters, action, loss, epsilon, reward, torch.max(prediction)))
        writer.add_scalar('Train/Loss', loss, iter)
        writer.add_scalar('Train/Epsilon', epsilon, iter)
        writer.add_scalar('Train/Reward', reward, iter)
        writer.add_scalar('Train/Q-Value', torch.max(prediction), iter)
        if (iter+1) % 1000000 == 0:
            torch.save(model, "{}/flappy_bird_{}".format(arguments.saved_path, iter+1))

    torch.save(model, "{}/flappy_bird".format(arguments.saved_path))


if __name__ == '__main__':
    args = get_args()
    training(args)
