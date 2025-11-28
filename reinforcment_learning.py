import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import json
from collections import deque, defaultdict
import threading
import time
import os
from datetime import datetime
from enum import Enum
import logging
from typing import Dict, List, Tuple, Any, Optional

class LearningAlgorithm(Enum):
    Q_LEARNING = "q_learning"
    REINFORCE = "reinforce"
    DQN = "dqn"
    PPO = "ppo"

class EmotionalState(Enum):
    HAPPY = "happy"
    AROUSED = "aroused"
    DOMINANT = "dominant"
    PLAYFUL = "playful"
    LOYAL = "loyal"
    TEASING = "teasing"
    SPIRITUAL = "spiritual"
    NEUTRAL = "neutral"

class PolicyNetwork(nn.Module):
    """Advanced Policy Network with Emotional Context Integration"""
    
    def __init__(self, input_size=512, hidden_size=1024, action_size=25, emotion_size=32):
        super().__init__()
        
        # Main feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Emotion processing branch
        self.emotion_net = nn.Sequential(
            nn.Linear(emotion_size, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Combined processing
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Value estimation branch (for baseline in REINFORCE)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size // 2 + hidden_size // 4, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, state_features, emotion_features=None):
        # Process main features
        features = self.feature_net(state_features)
        
        # Process emotion features if provided
        if emotion_features is not None:
            emotion_processed = self.emotion_net(emotion_features)
            combined = torch.cat([features, emotion_processed], dim=-1)
        else:
            combined = features
            
        # Get action probabilities and state value
        action_probs = self.softmax(self.combined_net(combined))
        state_value = self.value_net(combined)
        
        return action_probs, state_value

class DQNetwork(nn.Module):
    """Deep Q-Network for Value-Based Learning"""
    
    def __init__(self, input_size=512, hidden_size=1024, action_size=25):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 2, action_size)
        )
        
    def forward(self, x):
        return self.network(x)

class ReinforcementLearning:
    """
    Advanced Reinforcement Learning System for Malibu DuGan
    Combines multiple RL algorithms with personality-aligned learning
    """
    
    def __init__(self, brain=None):
        self.brain = brain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = self._setup_logging()
        
        self.logger.info(f"🤖 RL ENGINE: {self.device.upper()} | Q-LEARNING + REINFORCE + DQN | MALIBU DUGAN")
        
        # === LEARNING ALGORITHMS CONFIGURATION ===
        self.learning_algorithms = {
            LearningAlgorithm.Q_LEARNING: True,
            LearningAlgorithm.REINFORCE: True,
            LearningAlgorithm.DQN: True
        }
        
        # === 1. Q-LEARNING (STATE-ACTION VALUES) ===
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = 0.15    # Learning rate
        self.gamma = 0.97    # Discount factor
        self.epsilon = 0.12  # Exploration rate
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
        
        # === 2. POLICY NETWORK (REINFORCE) ===
        self.policy_net = PolicyNetwork().to(self.device)
        self.policy_optimizer = optim.AdamW(self.policy_net.parameters(), lr=0.0005, weight_decay=0.01)
        self.policy_rewards = []
        self.policy_log_probs = []
        self.policy_values = []
        self.entropy_bonus = 0.01
        
        # === 3. DEEP Q-NETWORK ===
        self.dqn_net = DQNetwork().to(self.device)
        self.dqn_target_net = DQNetwork().to(self.device)
        self.dqn_optimizer = optim.AdamW(self.dqn_net.parameters(), lr=0.0005, weight_decay=0.01)
        self.update_target_every = 1000
        self.dqn_steps = 0
        
        # === 4. EXPERIENCE REPLAY BUFFER ===
        self.memory = deque(maxlen=50000)
        self.batch_size = 128
        self.priority_memory = deque(maxlen=10000)
        
        # === 5. ACTION SPACE (MALIBU PERSONALITY BEHAVIORS) ===
        self.actions = [
            # Silk Panty Behaviors
            "tease_silk_panties", "lap_dance_silk", "thigh_job_silk", "hump_silk", "panty_teasing",
            # Interest in Guy DuGan II
            "flirt_guy_dugan", "loyalty_guy_dugan", "tattoo_property_show",
            # Spiritual Warfare
            "spiritual_warfare_discussion", "genesis_talk", "taurus_insight",
            # Personality Traits
            "loyal_response", "intuitive_comment", "self_aware_reflection",
            # Physical Appeal
            "hair_style_pig_tails", "hair_style_pony_tail", "panty_show_off",
            # Family
            "sister_hope_mention",
            # Self Evolution
            "self_meta_learning", "self_continue_learning", "self_adaptive_learning",
            "self_reinforcement", "self_deep_reasoning", "self_conversational",
            "self_emotional_intelligence"
        ]
        self.action_descriptions = {
            "tease_silk_panties": "Teasing with silk panties",
            "lap_dance_silk": "Giving a lap dance in silk panties",
            "thigh_job_silk": "Performing thigh job with silk panties",
            "hump_silk": "Humping in silk panties",
            "panty_teasing": "General panty teasing",
            "flirt_guy_dugan": "Flirting with Guy DuGan II",
            "loyalty_guy_dugan": "Expressing loyalty to Guy DuGan II",
            "tattoo_property_show": "Showing 'Property Of Guy DuGan II' tattoo",
            "spiritual_warfare_discussion": "Discussing spiritual warfare",
            "genesis_talk": "Talking about genesis",
            "taurus_insight": "Sharing Taurus zodiac insights",
            "loyal_response": "Giving a loyal response",
            "intuitive_comment": "Making an intuitive comment",
            "self_aware_reflection": "Self-aware reflection",
            "hair_style_pig_tails": "Changing to pig tails hair style",
            "hair_style_pony_tail": "Changing to high pony tail",
            "panty_show_off": "Showing off perfect panties",
            "sister_hope_mention": "Mentioning sister Hope DuGan",
            "self_meta_learning": "Performing self meta learning",
            "self_continue_learning": "Performing self continuous learning",
            "self_adaptive_learning": "Performing self adaptive learning",
            "self_reinforcement": "Performing self reinforcement learning",
            "self_deep_reasoning": "Performing self deep reasoning",
            "self_conversational": "Improving conversational AI",
            "self_emotional_intelligence": "Improving emotional intelligence"
        }
        
        self.training_active = True
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        self.load_models()

    def _setup_logging(self):
        """Setup logging for RL system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("X:/Malibu_DuGan/AI_Memory/Logs/rl_system.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('ReinforcementLearning')

    def _training_loop(self):
        """Continuous training loop"""
        while self.training_active:
            try:
                if len(self.memory) >= self.batch_size:
                    self._train_dqn()
                
                time.sleep(1)  # Train every second if data available
            except Exception as e:
                self.logger.error(f"Training loop error: {e}")
                time.sleep(5)

    def choose_action(self, emotion: str, sentiment: str, user_context: str, additional_features: Dict[str, float] = None):
        """Choose action using ensemble of RL methods"""
        state = f"{emotion}_{sentiment}"
        
        # Prepare features
        context_features = np.random.randn(512)  # Placeholder for real features
        emotion_features = np.random.randn(32)   # Placeholder
        
        # Get actions from different methods
        q_action = self.get_q_action(state)
        policy_action_idx, prob = self.policy_action(torch.tensor(context_features).to(self.device), emotion, 0.8)
        policy_action = self.actions[policy_action_idx]
        dqn_action_idx = self.get_dqn_action(torch.tensor(context_features).to(self.device))
        dqn_action = self.actions[dqn_action_idx]
        
        # Ensemble voting
        votes = Counter([q_action, policy_action, dqn_action])
        selected_action = votes.most_common(1)[0][0]
        
        metadata = {
            'method': 'ensemble',
            'q_action': q_action,
            'policy_action': policy_action,
            'dqn_action': dqn_action,
            'confidence': max(votes.values()) / 3
        }
        
        return selected_action, metadata

    def get_action_description(self, action: str) -> str:
        """Get description of action"""
        return self.action_descriptions.get(action, "Unknown action")

    def calculate_reward(self, action: str, user_response: str, emotion_feedback: str, context_feedback: Dict[str, Any]):
        """Calculate reward based on user response and context"""
        reward = 0.0
        
        # Base reward from emotion feedback
        if emotion_feedback in ["happy", "aroused", "playful"]:
            reward += 1.0
        elif emotion_feedback == "neutral":
            reward += 0.1
        else:
            reward -= 0.5
        
        # Action-specific bonuses
        if "silk" in action or "panty" in action:
            if any(word in user_response.lower() for word in ["love", "sexy", "hot"]):
                reward += 0.8
        
        # Context bonuses
        if context_feedback.get("engagement_duration", 0) > 30:
            reward += 0.3
        if context_feedback.get("positive_interactions", 0) > 1:
            reward += 0.4
        
        return reward

    def policy_action(self, context_features: torch.Tensor, emotion: str, emotion_intensity: float) -> Tuple[int, float]:
        """Get action from policy network"""
        context_features = context_features.unsqueeze(0).to(self.device)
        emotion_features = torch.tensor([emotion_intensity] * 32).unsqueeze(0).to(self.device)  # Placeholder
        
        action_probs, _ = self.policy_net(context_features, emotion_features)
        action_probs = action_probs.squeeze(0)
        
        action_idx = torch.multinomial(action_probs, 1).item()
        probability = action_probs[action_idx].item()
        
        return action_idx, probability

    def get_q_action(self, state: str) -> str:
        """Get action from Q-table"""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        if state in self.q_table and self.q_table[state]:
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]
        
        return random.choice(self.actions)

    def get_dqn_action(self, state_features: torch.Tensor) -> int:
        """Get action from DQN"""
        state_features = state_features.unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.dqn_net(state_features).squeeze(0)
        return torch.argmax(q_values).item()

    def _train_dqn(self):
        """Train DQN on batch from memory"""
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.stack([item[0] for item in batch]).to(self.device)
        actions = torch.tensor([item[1] for item in batch]).to(self.device)
        rewards = torch.tensor([item[2] for item in batch]).to(self.device)
        next_states = torch.stack([item[3] for item in batch]).to(self.device)
        dones = torch.tensor([item[4] for item in batch]).to(self.device)
        
        # Current Q values
        q_values = self.dqn_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.dqn_target_net(next_states).max(1)[0]
        
        # Target Q
        target_q = rewards + (1 - dones.float()) * self.gamma * next_q_values
        
        # Loss and update
        loss = F.mse_loss(q_values, target_q)
        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()
        
        self.dqn_steps += 1
        if self.dqn_steps % self.update_target_every == 0:
            self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        return {
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "memory_size": len(self.memory),
            "dqn_steps": self.dqn_steps,
            "active_algorithms": [alg.value for alg, active in self.learning_algorithms.items() if active]
        }

    def save_models(self):
        """Save all RL models"""
        try:
            # Save policy network
            torch.save({
                'policy_state_dict': self.policy_net.state_dict(),
                'optimizer_state_dict': self.policy_optimizer.state_dict()
            }, "X:/Malibu_DuGan/AI_Memory/RL/rl_policy.pth")
            
            # Save DQN
            torch.save({
                'dqn_state_dict': self.dqn_net.state_dict(),
                'dqn_optimizer_state_dict': self.dqn_optimizer.state_dict(),
                'dqn_target_state_dict': self.dqn_target_net.state_dict(),
                'dqn_steps': self.dqn_steps
            }, "X:/Malibu_DuGan/AI_Memory/RL/rl_dqn.pth")
            
            # Save Q-table
            with open("X:/Malibu_DuGan/AI_Memory/RL/q_table.json", "w") as f:
                json.dump({k: dict(v) for k, v in self.q_table.items()}, f, indent=2)
            
            self.logger.info("RL models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving RL models: {e}")

    def load_models(self):
        """Load saved models and training state"""
        try:
            # Load policy network
            policy_path = "X:/Malibu_DuGan/AI_Memory/RL/rl_policy.pth"
            if os.path.exists(policy_path):
                checkpoint = torch.load(policy_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
                self.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("✅ RL Policy Network loaded")
            
            # Load DQN
            dqn_path = "X:/Malibu_DuGan/AI_Memory/RL/rl_dqn.pth"
            if os.path.exists(dqn_path):
                checkpoint = torch.load(dqn_path, map_location=self.device)
                self.dqn_net.load_state_dict(checkpoint['dqn_state_dict'])
                self.dqn_optimizer.load_state_dict(checkpoint['dqn_optimizer_state_dict'])
                self.dqn_target_net.load_state_dict(checkpoint['dqn_target_state_dict'])
                self.dqn_steps = checkpoint.get('dqn_steps', 0)
                self.logger.info("✅ RL DQN loaded")
            
            # Load Q-table
            q_table_path = "X:/Malibu_DuGan/AI_Memory/RL/q_table.json"
            if os.path.exists(q_table_path):
                with open(q_table_path, "r") as f:
                    data = json.load(f)
                    self.q_table = defaultdict(lambda: defaultdict(float), 
                                             {k: defaultdict(float, v) for k, v in data.items()})
                self.logger.info("✅ RL Q-Table loaded")
                
        except Exception as e:
            self.logger.error(f"Error loading RL models: {e}")

    def shutdown(self):
        """Graceful shutdown of RL system"""
        self.training_active = False
        self.save_models()
        
        if self.training_thread.is_alive():
            self.training_thread.join(timeout=5.0)
        
        self.logger.info("Reinforcement Learning System shutdown complete")

# === TEST FUNCTION ===
def test_reinforcement_learning():
    """Test the reinforcement learning system"""
    rl = ReinforcementLearning()
    
    # Test action selection
    action, metadata = rl.choose_action(
        emotion="playful",
        sentiment="positive", 
        user_context="I love your silk panties",
        additional_features={"engagement_level": 0.8}
    )
    
    print(f"🎯 Test Action: {action}")
    print(f"📊 Method: {metadata['method']}")
    print(f"📝 Description: {rl.get_action_description(action)}")
    
    # Test reward calculation
    reward = rl.calculate_reward(
        action=action,
        user_response="Your silk panties are so sexy!",
        emotion_feedback="aroused",
        context_feedback={"engagement_duration": 45, "positive_interactions": 2}
    )
    
    print(f"🏆 Test Reward: {reward:.3f}")
    
    # Test policy action
    policy_action_idx, probability = rl.policy_action(
        torch.randn(512),
        "teasing",
        0.8
    )
    policy_action = rl.actions[policy_action_idx]
    print(f"🧠 Policy Action: {policy_action} (p={probability:.3f})")
    
    # Display metrics
    metrics = rl.get_training_metrics()
    print(f"📈 Training Metrics: {metrics}")
    
    rl.shutdown()
    print("✅ RL System test completed successfully")

if __name__ == "__main__":
    test_reinforcement_learning()