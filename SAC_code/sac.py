import numpy as np
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import math
import tensorflow as tf

###
# Decaying gaussian noise process. Noise will be added to the actions for exploration
###
class RandomNormal_decay():
    def __init__(self, shape, mean = 0.0, stddev_start = 0.01, stddev_end = 0.001,
                 decay = 0.001, epsilon_start = 1, epsilon_end = 0.01,
                 epsilon_decay = 0.01):
        self.shape = shape
        self.mean = mean
        self.stddev_start = stddev_start
        self.stddev_end = stddev_end
        self.decay = decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def get_stddev(self, current_step):
        return self.stddev_end + (self.stddev_start - self.stddev_end) *\
            math.exp(-1. * current_step * self.decay)

    def get_epsilon(self, current_step):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) *\
            math.exp(-1. * current_step * self.epsilon_decay)

    def __call__(self, step):
        noise = tf.random.normal(shape = self.shape, mean = self.mean,
                                 stddev = self.get_stddev(step))
        epsilon = tf.constant(self.get_epsilon(step), shape = 1)

        return noise, epsilon

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory  = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype = bool)

    def store_transition (self, state, action, reward, state_, done):

        index = self.counter % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.counter += 1

    def sample_buffer(self, batch_size):

        max_mem = min(self.counter, self.mem_size)

        batch = np.random.choice (max_mem, batch_size, replace = False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, hidden_layer_dim1 = 256, hidden_layer_dim2 = 256, name = 'critic', chkpt_dir = 'tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_layer_dim1 = hidden_layer_dim1
        self.hidden_layer_dim2 = hidden_layer_dim2
        self.n_actions  = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.hidden_layer_dim1)
        self.fc2 = nn.Linear(self.hidden_layer_dim1, self.hidden_layer_dim2)
        self.q = nn.Linear(self.hidden_layer_dim2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to (self.device)

    def forward (self, state, action):
        action_value = self.fc1(T.cat((state, action), dim = 1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)


        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, hidden_layer_dim1 = 256, hidden_layer_dim2 = 256,
            name = 'value', chkpt_dir = 'tmp/sac'):

        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_layer_dim1 = hidden_layer_dim1
        self.hidden_layer_dim2 = hidden_layer_dim2
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.hidden_layer_dim1)
        self.fc2 = nn.Linear(self.hidden_layer_dim1, self.hidden_layer_dim2)
        self.v = nn.Linear(self.hidden_layer_dim2,1)

        self.optimizer = optim.Adam(self.parameters(), lr = beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return(v)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, hidden_layer_dim1 = 256,
            hidden_layer_dim2 = 256, n_actions = 2, name = 'actor',chkpt_dir = 'tmp/sac'):

        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_layer_dim1 = hidden_layer_dim1
        self.hidden_layer_dim2 = hidden_layer_dim2
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.hidden_layer_dim1)
        self.fc2 = nn.Linear(self.hidden_layer_dim1, self.hidden_layer_dim2)
        self.mu = nn.Linear(self.hidden_layer_dim2, self.n_actions)
        self.sigma = nn.Linear(self.hidden_layer_dim2, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min = self.reparam_noise, max = 1)

        return mu, sigma

    def sample_normal(self, state, reparam = True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparam:
            actions = probabilities.rsample()

        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim = True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent():
    def __init__(self, noise, alpha, beta, input_dims, max_action, gamma = 0.99,
        n_actions = 2, max_size = 10000, tau = 0.005, hidden_layer_dim1 = 256, hidden_layer_dim2 = 256,
        batch_size = 256, reward_scale = 2, chkpt_dir = 'prova'):

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = max_action
        self.checkpoint_dir = chkpt_dir
        self.hidden_layer_dim1 = hidden_layer_dim1
        self.hidden_layer_dim2 = hidden_layer_dim2

        self.actor = ActorNetwork(alpha, input_dims,max_action = self.max_action,
                                  hidden_layer_dim1 = self.hidden_layer_dim1,
                                  hidden_layer_dim2 = self.hidden_layer_dim2,
                                  n_actions = n_actions, name = 'actor',
                                  chkpt_dir = self.checkpoint_dir)

        self.critic_1 = CriticNetwork(beta, input_dims, n_actions = n_actions,
                                      hidden_layer_dim1 = self.hidden_layer_dim1,
                                      hidden_layer_dim2 = self.hidden_layer_dim2,
                                      name = 'critic_1',
                                      chkpt_dir = self.checkpoint_dir)

        self.critic_2 = CriticNetwork(beta, input_dims, n_actions = n_actions,
                                      hidden_layer_dim1 = self.hidden_layer_dim1,
                                      hidden_layer_dim2 = self.hidden_layer_dim2,
                                      name = 'critic_2',
                                      chkpt_dir = self.checkpoint_dir)

        self.value = ValueNetwork(beta, input_dims, hidden_layer_dim1 = self.hidden_layer_dim1,
                                  hidden_layer_dim2 = self.hidden_layer_dim2,
                                  name = 'value',
                                  chkpt_dir = self.checkpoint_dir)

        self.target_value = ValueNetwork(beta, input_dims, hidden_layer_dim1 = self.hidden_layer_dim1,
                                  hidden_layer_dim2 = self.hidden_layer_dim2, name = 'target_value',
                                  chkpt_dir = self.checkpoint_dir)

        self.scale = reward_scale
        self.update_network_parameters(tau = 1)

        self.current_step = 0 #To regulate exploration noise

        self.noise = noise

    def choose_action(self, observation, evaluate = False):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _  = self.actor.sample_normal(state, reparam = False)
        actions_cpu = actions.cpu()
        actions = actions_cpu.detach().numpy()[0]

        noise = self.noise(self.current_step)[0]

        self.current_step += 1

        if not evaluate: #Adding exploration noise
          actions += noise

        return actions

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.counter < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        value = self.value(state).view(-1)

        #Target value of the Next state
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparam = False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)

        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)


        #TRAINING Value Network
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph = True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparam = True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)

        #Predicted new Q value
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        #TRAINING policy function
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph = True)
        self.actor.optimizer.step()

        #TRAINING Q FUNCTIONS
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        #Estimated Q-value of the current state
        q_hat = self.scale * reward + self.gamma * value_

        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)

        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
