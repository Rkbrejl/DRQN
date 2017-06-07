import itertools as it
from time import time, sleep
import numpy as np
import skimage.color
import skimage.transform
import tensorflow as tf
from tqdm import trange
from vizdoom import *

from Agent import Agent

FRAME_REPEAT = 4 # How many frames 1 action should be repeated
UPDATE_FREQUENCY = 4 # How many actions should be taken between each network update

RESOLUTION = (80, 45, 3) # Resolution
BATCH_SIZE = 32 # Batch size for experience replay
LEARNING_RATE = 0.00025 # Learning rate of model
GAMMA = 0.99 # Discount factor

MEMORY_CAP = 1000000 # Amount of samples to store in memory

EPSILON_MAX = 1 # Max exploration rate
EPSILON_MIN = 0.1 # Min exploration rate
EPSILON_DECAY_STEPS = 2e5 # How many steps to decay from max exploration to min exploration

RANDOM_WANDER_STEPS = 50000 # How many steps to be sampled randomly before training starts

TRACE_LENGTH = 8 # How many traces are used for network updates
HIDDEN_SIZE = 768 # Size of the third convolutional layer when flattened

EPOCHS = 200 # Epochs for training (1 epoch = 10k training steps and 10 test episodes)
STEPS_PER_EPOCH = 10000 # How actions to be taken per epoch
EPISODES_TO_TEST = 10 # How many test episodes to be run per epoch for logging performance
EPISODE_TO_WATCH = 10 # How many episodes to watch after training is complete

TAU = 0.001 # How much the target network should be updated towards the online network at each update

LOAD_MODEL = True # Load a saved model?
SAVE_MODEL = False # Save a model while training?
SKIP_LEARNING = True # Skip training completely and just watch?

scenario_path = "../../ViZDoom/scenarios/my_way_home.cfg" # Name and path of scenario
model_savefile = "Models/MWH/model" # Name and path of the model
reward_savefile = "Rewards_MWH.txt"

##########################################

def initialize_vizdoom():
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(scenario_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_400X225)
    game.init()

    print("Doom initialized.")
    return game

def preprocess(img):
    img = skimage.transform.resize(img,RESOLUTION)
    img = img.astype(np.float32)
    return img

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def saveScore(score):
    my_file = open(reward_savefile, 'a')  # Name and path of the reward text file
    my_file.write("%s\n" % test_scores.mean())
    my_file.close()

###########################################

game = initialize_vizdoom()

n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]
ACTION_COUNT = len(actions)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

SESSION = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

agent = Agent(memory_cap = MEMORY_CAP, batch_size = BATCH_SIZE, resolution = RESOLUTION, action_count = ACTION_COUNT,
            session = SESSION, lr = LEARNING_RATE, gamma = GAMMA, epsilon_min = EPSILON_MIN, trace_length=TRACE_LENGTH,
            epsilon_decay_steps = EPSILON_DECAY_STEPS, epsilon_max=EPSILON_MAX, hidden_size=HIDDEN_SIZE)

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, TAU)

if LOAD_MODEL:
    print("Loading model from: ", model_savefile)
    saver.restore(SESSION, model_savefile)
else:
    init = tf.global_variables_initializer()
    SESSION.run(init)

##########################################

if not SKIP_LEARNING:
    time_start = time()
    print("\nFilling out replay memory")
    updateTarget(targetOps, SESSION)

    episode_buffer = []
    agent.reset_cell_state()
    state = preprocess(game.get_state().screen_buffer)
    for _ in trange(RANDOM_WANDER_STEPS, leave=False):
        action = agent.random_action()
        reward = game.make_action(actions[action], FRAME_REPEAT)
        done = game.is_episode_finished()
        if not done:
            state_new = preprocess(game.get_state().screen_buffer)
        else:
            state_new = None

        agent.add_transition(state, action, reward, state_new, done)
        state = state_new

        if done:
            game.new_episode()
            agent.reset_cell_state()
            state = preprocess(game.get_state().screen_buffer)

    for epoch in range(EPOCHS):
        print("\n\nEpoch %d\n-------" % (epoch + 1))

        train_episodes_finished = 0
        train_scores = []

        print("Training...")
        game.new_episode()

        episode_buffer = []
        agent.reset_cell_state()
        state = preprocess(game.get_state().screen_buffer)
        for learning_step in trange(STEPS_PER_EPOCH, leave=False):
            action = agent.act(state)
            reward = game.make_action(actions[action], FRAME_REPEAT)
            done = game.is_episode_finished()
            if not done:
                state_new = preprocess(game.get_state().screen_buffer)
            else:
                state_new = None

            agent.add_transition(state, action, reward, state_new, done)
            state = state_new

            if learning_step % UPDATE_FREQUENCY == 0:
                agent.learn_from_memory()
                updateTarget(targetOps, SESSION)

            if done:
                train_scores.append(game.get_total_reward())
                train_episodes_finished += 1
                game.new_episode()
                agent.reset_cell_state()
                state = preprocess(game.get_state().screen_buffer)

        print("%d training episodes played." % train_episodes_finished)
        train_scores = np.array(train_scores)

        print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()),
            "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        print("\nTesting...")

        test_scores = []
        for test_step in trange(EPISODES_TO_TEST, leave=False):
            game.new_episode()
            agent.reset_cell_state()
            while not game.is_episode_finished():
                state = preprocess(game.get_state().screen_buffer)
                action = agent.act(state, train=False)
                game.make_action(actions[action], FRAME_REPEAT)
            test_scores.append(game.get_total_reward())

        test_scores = np.array(test_scores)
        print("Results: mean: %.1f±%.1f," % (test_scores.mean(), test_scores.std()),
              "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())

        if SAVE_MODEL:
            saveScore(test_scores.mean())
            saver.save(SESSION, model_savefile)
            print("Saving the network weigths to:", model_savefile)
            if epoch % (EPOCHS/5) == 0 and epoch is not 0:
                saver.save(SESSION, model_savefile, global_step=epoch)

        print("Total ellapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

print("TIME TO WATCH!!")
# Reinitialize the game with window visible
game.close()
game.set_window_visible(True)
game.set_mode(Mode.ASYNC_PLAYER)
game.init()
score = []
for _ in trange(EPISODE_TO_WATCH, leave=False):
    game.new_episode()
    agent.reset_cell_state()
    while not game.is_episode_finished():
        state = preprocess(game.get_state().screen_buffer)
        action = agent.act(state, train=False)
        game.set_action(actions[action])
        for i in range(FRAME_REPEAT):
            game.advance_action()
            done = game.is_episode_finished()
            if done:
                break

    # Sleep between episodes
    sleep(1.0)
    score.append(game.get_total_reward())
score = np.array(score)
game.close()
print("Results: mean: %.1f±%.1f," % (score.mean(), score.std()),
          "min: %.1f" % score.min(), "max: %.1f" % score.max())
