import logging

from PIL import Image
import keras
import numpy as np
import imageio
import tensorflow as tf
from tf_agents.agents.dqn.dqn_agent import  DqnAgent
from tf_agents.environments import suite_gym
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories.trajectory import to_transition
from tf_agents.utils.common import function

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

class ShowProgress:
    """
    Class that implements  progress reports for the buffer
    """
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")

VIDEO_FILENAME = "result"
N_ITERATIONS = 50000
REPLAY_BUFFER_SIZE = 20000

# creating the environment
max_episode_steps = 27000
environment_name = "BreakoutNoFrameskip-v4"

env = suite_gym.load(
    environment_name,
    max_episode_steps = max_episode_steps,
    gym_env_wrappers = [AtariPreprocessing, FrameStack4]
)

tf_env = TFPyEnvironment(env)


# creating the deep Q-network
preprocessing_layer = keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)

conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params = [512]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers = preprocessing_layer,
    conv_layer_params = conv_layer_params,
    fc_layer_params = fc_layer_params
)

# creating the DQN agent

train_step = tf.Variable(0)
update_period = 4 #train the model every 4 steps
optimizer = tf.keras.optimizers.RMSprop(learning_rate = 2.5e-4, rho=0.95, momentum=0.0, epsilon = 0.00001, centered = True)

epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate = 1.0,
    decay_steps = 250000 // update_period,
    end_learning_rate=0.01
)

agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network = q_net,
                 optimizer = optimizer,
                 target_update_period = 2000,
                 td_errors_loss_fn = keras.losses.Huber(reduction = "none"),
                 gamma = 0.99,
                 train_step_counter = train_step,
                 epsilon_greedy = lambda: epsilon_fn(train_step))

agent.initialize()


# creating the replay buffer and observer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = agent.collect_data_spec,
    batch_size = tf_env.batch_size,
    max_length = REPLAY_BUFFER_SIZE
)

replay_buffer_observer = replay_buffer.add_batch

# creating training metrics
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric()
]

avgReward = tf_metrics.AverageReturnMetric()

# creating the collect driver
collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers = [replay_buffer_observer, avgReward],
    num_steps = update_period #collect 4 steps for each training iteration
)

# creating a random policy and driver to warm up the replay buffer
initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                        tf_env.action_spec())

init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers = [replay_buffer.add_batch, ShowProgress(20000)],
    num_steps = 20000
)

final_time_step, final_policy_state = init_driver.run()

# creating the dataset
trajectories, buffer_info = replay_buffer.get_next(
    sample_batch_size = 2, num_steps = 3
)

time_steps, action_steps, next_time_steps = to_transition(trajectories)

dataset = replay_buffer.as_dataset(
    sample_batch_size = 64,
    num_steps = 2,
    num_parallel_calls = 3).prefetch(3)


# creating the training loop
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)

    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)

        print("\r{} loss:{:.5f}".format(iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            print(avgReward.result())

# begin training for 50000 iterations
train_agent(N_ITERATIONS)


def create_policy_eval_video(filename, num_episodes=1, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = tf_env.reset()
            video.append_data(tf_env.render().numpy()[0])

            while not time_step.is_last():
                action_step = agent.policy.action(time_step)
                time_step = tf_env.step(action_step)
                video.append_data(tf_env.render().numpy()[0])

create_policy_eval_video(VIDEO_FILENAME)