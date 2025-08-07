import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models

class RLAgent:
    """Reinforcement Learning Agent with Deep Q-Network"""
    
    def __init__(self, state_size, action_size, agent_type):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_type = agent_type  # 'red', 'blue', or 'white'
        
        # Hyperparameters
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory = deque(maxlen=2000)
        
        # Neural network model
        self.model = self._build_model()
        
        # Target network for stable learning
        self.target_model = self._build_model()
        self.update_target_frequency = 10
        
        # Curriculum learning parameters
        self.curriculum_level = 1
        self.success_threshold = 0.8
        self.episode_rewards = []
        
    def _build_model(self):
        """Build the neural network model"""
        model = models.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            loss='huber_loss',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose an action based on the current state"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train the agent using experiences from memory"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([transition[0] for transition in minibatch])
        states = states.reshape(len(minibatch), self.state_size)
        
        next_states = np.array([transition[3] for transition in minibatch])
        next_states = next_states.reshape(len(minibatch), self.state_size)
        
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(next_q_values[i])
            
            q_values[i][action] = target
        
        self.model.fit(states, q_values, epochs=1, verbose=0)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Update the target network with the main network's weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def evaluate_curriculum(self, success_rate):
        """Evaluate if we should advance to the next curriculum level"""
        if success_rate > self.success_threshold:
            self.curriculum_level += 1
            self._adjust_difficulty()
            return True
        return False
    
    def _adjust_difficulty(self):
        """Adjust the difficulty based on curriculum level"""
        # Implement difficulty adjustment logic based on curriculum level
        if self.curriculum_level > 3:
            # Increase complexity for red team agents
            if self.agent_type == 'red':
                self.gamma = min(0.99, self.gamma + 0.01)
                self.epsilon_min = max(0.001, self.epsilon_min - 0.001)
            # Different adjustments for blue and white team agents
            elif self.agent_type == 'blue':
                self.learning_rate = min(0.005, self.learning_rate + 0.0005)
            elif self.agent_type == 'white':
                self.batch_size = min(64, self.batch_size + 8)
    
    def get_training_metrics(self):
        """Get metrics for training evaluation"""
        if not self.episode_rewards:
            return {}
        
        avg_reward = np.mean(self.episode_rewards[-100:])
        recent_reward = self.episode_rewards[-1] if self.episode_rewards else 0
        
        return {
            'curriculum_level': self.curriculum_level,
            'epsilon': self.epsilon,
            'average_reward': avg_reward,
            'recent_reward': recent_reward,
            'memory_size': len(self.memory),
            'learning_rate': self.learning_rate
        }
    
    def save_model(self, path):
        """Save the model to disk"""
        self.model.save(f"{path}_{self.agent_type}_agent.h5")
    
    def load_model(self, path):
        """Load a pre-trained model"""
        self.model = models.load_model(f"{path}_{self.agent_type}_agent.h5")
        self.target_model = models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

# Example usage
if __name__ == "__main__":
    # Example of creating and training an RL agent
    state_size = 10  # Number of state features
    action_size = 4  # Number of possible actions
    
    # Create a red team agent
    red_agent = RLAgent(state_size, action_size, 'red')
    
    # Simulated training loop
    for episode in range(1000):
        # Simulated state
        state = np.random.rand(1, state_size)
        
        # Choose action
        action = red_agent.act(state)
        
        # Simulated environment response
        next_state = np.random.rand(1, state_size)
        reward = np.random.rand() * 2 - 1  # Random reward between -1 and 1
        done = False
        
        # Store experience
        red_agent.remember(state, action, reward, next_state, done)
        
        # Learn from experiences
        red_agent.replay()
        
        # Periodically update target model
        if episode % red_agent.update_target_frequency == 0:
            red_agent.update_target_model()
        
        # Track rewards
        red_agent.episode_rewards.append(reward)
        
        # Evaluate curriculum progress
        if episode % 100 == 0:
            success_rate = np.mean(red_agent.episode_rewards[-100:])
            red_agent.evaluate_curriculum(success_rate)
            metrics = red_agent.get_training_metrics()
            print(f"Episode {episode}, Metrics: {metrics}")
    
    # Save trained model
    red_agent.save_model("xorbfw_models")
    
    print("Training complete. Model saved.")

# This implementation provides:
# 1. A flexible RL agent framework that can be used for all agent types
# 2. Curriculum learning capabilities that adapt difficulty based on performance
# 3. Experience replay for efficient learning
# 4. Target network for stable Q-learning
# 5. Training metrics tracking and model persistence
# 6. Adaptive learning parameters based on agent type and curriculum level
# 
# The example usage demonstrates how the agent would be trained in a simulation environment.