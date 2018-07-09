from keras import layers, models, optimizers, initializers, regularizers
from keras import backend as K
import numpy as np
import copy
from agents.OUNoise import OUNoise
from agents.ExperienceReplay import ExperienceReplayBuffer

class Actor:
    def __init__(self, state_space, action_space, action_min, action_max, hidden_units, learning_rate, q_lambda):
        self.state_space = state_space
        self.action_space = action_space
        self.action_max = action_max
        self.action_min = action_min
        self.action_range = action_max - action_min
        self.learning_rate = learning_rate
        self.q_lambda = q_lambda
        
        # Neural Network definition.
        
        # Batch normalization on the state input and all layers of U network.
        # var_wi = initializers.VarianceScaling(scale=1.0/3.0, mode='fan_in', distribution='uniform')
        var_wi = initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')
        out_wi = initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
        
        # Input
        input_states = layers.Input(shape=(self.state_space,), name='input_states')
        
        # Layer 1
        layer = layers.Dense(units=400, kernel_regularizer=regularizers.l2(self.q_lambda))(input_states)
        layer = layers.BatchNormalization()(layer)
        layer = layers.Activation('relu')(layer)
        
        # Layer 2
        layer = layers.Dense(units=300, kernel_regularizer=regularizers.l2(self.q_lambda))(layer)
        layer = layers.BatchNormalization()(layer)       
        layer = layers.Activation('relu')(layer)
        
        # Output between 0,1.
        norm_action = layers.Dense(self.action_space, kernel_initializer=out_wi, activation='sigmoid',
                                   name='norm_action')(layer)
        
        # Adapt actions for the range in which rotors work.
        actions = layers.Lambda(lambda x: (x*self.action_range) + self.action_min, name='actions')(norm_action)
        ############## 
        
        # Instantiate Model.
        self.model = models.Model(input=input_states, output=actions)
        
        # Define Loss
        action_gradients = layers.Input(shape=(self.action_space,))
        loss = K.mean(-action_gradients*actions)
        
        # Get trainable parameters and define backprop optimization.
        adam_optimizer = optimizers.Adam(lr=self.learning_rate)
        train_param = adam_optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        
        # keras.backend.learning_phase() gives a flag to be passed as input
        # to any Keras function that uses a different behavior at train time and test time.
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], 
                                   outputs=[], 
                                   updates=train_param)
        
        
class Critic:
    def __init__(self, state_space, action_space, hidden_units, learning_rate, q_lambda):
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.q_lambda = q_lambda
        
        # Network Architecture.
        # var_wi = initializers.VarianceScaling(scale=1.0/3.0, mode='fan_in', distribution='uniform')
        var_wi = initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')
        out_wi = initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
    
        
        ## States network.        
        # Input States
        input_states = layers.Input(shape=(self.state_space,), name='input_states')

        # Layer 1.
        layer_states = layers.Dense(units=400, kernel_regularizer=regularizers.l2(self.q_lambda))(input_states)
        layer_states = layers.BatchNormalization()(layer_states)
        layer_states = layers.Activation('relu')(layer_states)
        
        # Layer 2.
        layer_states = layers.Dense(units=300, activation='relu',
                                    kernel_regularizer=regularizers.l2(self.q_lambda))(layer_states)
        
        
        ## Action network.
        # Input Actions
        input_actions = layers.Input(shape=(self.action_space,), name='input_actions') 

        # Layer 1.
        layer_actions = layers.Dense(units=300, activation='relu',
                                     kernel_regularizer=regularizers.l2(self.q_lambda))(input_actions)
        
        
        ## Advantage network.
        layer = layers.Add()([layer_states, layer_actions])
        layer = layers.Activation('relu')(layer)
        # layer = layers.Dense(units=hidden_units, kernel_initializer=var_wi, activation='relu',
        #                     kernel_regularizer=regularizers.l2(self.q_lambda))(layer)

        # Add final output layer to prduce action values (Q values)
        q_values = layers.Dense(units=1, kernel_initializer=out_wi, name='q_values')(layer)
        
        
        
        # Instantiate the model
        self.model = models.Model(inputs=[input_states, input_actions], outputs=q_values)
        
        # Optimizer and Loss.
        adam_optimizer = optimizers.Adam(lr=self.learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=adam_optimizer)
        
        # Define function to get action gradients.
        action_gradients = K.gradients(loss=q_values, variables=input_actions)
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients)
        
        
class DDPG_Agent:
    def __init__(self, task, noise, memory, rl_param, nn_hidden, actor_lr, critic_lr, q_lambda):
        # Adapted for this gym
        self.task = task
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.state_space = task.state_size
        self.action_space = task.action_size
        self.q_lambda = q_lambda
        
        # Instantiate Actors and Critics.
        self.actor = Actor(self.state_space, self.action_space, self.action_low, self.action_high, hidden_units=nn_hidden[0],
                           learning_rate=actor_lr, q_lambda=q_lambda)
        self.actor_target = Actor(self.state_space, self.action_space, self.action_low, self.action_high,
                                  hidden_units=nn_hidden[0], learning_rate=actor_lr, q_lambda=q_lambda)
        
        self.critic = Critic(self.state_space, self.action_space, hidden_units=nn_hidden[1], learning_rate=critic_lr, 
                             q_lambda=q_lambda)
        self.critic_target = Critic(self.state_space, self.action_space, hidden_units=nn_hidden[1], learning_rate=critic_lr, 
                             q_lambda=q_lambda)
        
        # Set same weights in target.
        self.actor_target.model.set_weights(self.actor.model.get_weights())
        self.critic_target.model.set_weights(self.critic.model.get_weights())
        
        # Noise for exploration.
        self.mean = noise[0]
        self.sigma = noise[1]
        self.theta = noise[2]
        self.ounoise = OUNoise(self.action_space, self.mean, self.sigma, self.theta)
        
        # Experience Replay memory.
        self.capacity = memory[0]
        self.batch_size = memory[1]
        self.er_buffer = ExperienceReplayBuffer(capacity=self.capacity, batch_size=self.batch_size)
        
        # RL parameters.
        self.gamma = rl_param[0]
        self.t = rl_param[1]
        
        # Keeping track of learning.
        self.learning_rewards = list()
        self.total_reward = None
        self.best_reward = -np.inf
        self.losses = list()
        
    def restart_task(self):
        if self.total_reward is not None:
            self.learning_rewards.append(self.total_reward)
            if self.total_reward > self.best_reward: self.best_reward = self.total_reward
        self.total_reward = 0
        state = self.task.reset()
        self.state = state
        self.ounoise.restart()
        return state
        
    def act(self, state, epsilon):
        self.action_wo_noise = self.actor.model.predict(np.reshape(state, newshape=(-1, self.state_space)))
        self.step_noise = self.ounoise.sample()*epsilon
        action = np.array(self.action_wo_noise[0] + self.step_noise[0]).reshape(-1, self.action_space)
        action_clipped = np.clip(a=action, a_min=self.action_low, a_max=self.action_high)
        return action_clipped
        
    # Saves expirience into memory and updates actor-critic weights.
    def store_learn(self, state, action, reward, done, next_state):
        
        # Store experience into exp replay memory.
        self.er_buffer.add_env_reaction((state, action, reward, done, next_state))
        
        # Learn if agent has enough experiences.
        if len(self.er_buffer.mem) > self.batch_size:
            self.learn()
        
        self.total_reward += reward
        # Update to the current state of the enviroment.
        self.state = next_state
     
    def soft_update(self):
        actor_current = np.array(self.actor.model.get_weights())
        critic_current = np.array(self.critic.model.get_weights())
        
        actor_target = np.array(self.actor_target.model.get_weights())
        critic_target = np.array(self.critic_target.model.get_weights())
        
        self.actor_target.model.set_weights(actor_target*(1-self.t) + self.t*actor_current)
        self.critic_target.model.set_weights(critic_target*(1-self.t) + self.t*critic_current)
    
    # Learn step of the agent, update weights of actor-critic and actor-critic target NN.
    def learn(self):
        states, actions, rewards, dones, next_states = self.er_buffer.sample_batch()
        states = np.vstack(states)
        actions = np.array(actions, dtype=np.float32).reshape(-1, self.action_space)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        dones = np.array(dones, dtype=np.uint8).reshape(-1, 1)
        next_states = np.vstack(next_states)
        
        # Get action for deterministic policy.
        next_actions = self.actor_target.model.predict_on_batch(next_states)
        next_q_values = self.critic_target.model.predict_on_batch([next_states, next_actions])
        
        # Need to handle the done case.
        targets = rewards + self.gamma*next_q_values*(1-dones)
        loss = self.critic.model.train_on_batch(x=[states, actions],y=targets)
        self.losses.append(loss)
        
        # Getting gradients before Critics backprop.
        action_gradients = self.critic.get_action_gradients([states, actions, 0])
        action_gradients_prev = action_gradients
        action_gradients = np.reshape(action_gradients[0], (-1, self.action_space))
        
        # Learning Phase = 0 (Test), we just want the gradient, no update on weights.
        self.actor.train_fn([states, action_gradients, 1])
        
        # Do soft update on weigths.
        self.soft_update()
        