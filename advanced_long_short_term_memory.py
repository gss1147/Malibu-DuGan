import sqlite3
import json
import yaml
import pickle
import os
import threading
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import re
import time
import copy
import random
import sys

# Add the AI_Python directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    print("✓ PyTorch loaded successfully")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not available - LSTM features disabled")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
    print("✓ scikit-learn loaded successfully")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠ scikit-learn not available - semantic features disabled")

class LSTMPredictor(nn.Module):
    """Enhanced LSTM model for response prediction and memory pattern learning"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Apply attention
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch, features)
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended = attended.transpose(0, 1)  # (batch, seq_len, features)
        
        # Global average pooling and max pooling
        avg_pool = attended.mean(dim=1)
        max_pool, _ = attended.max(dim=1)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        
        output = self.classifier(combined)
        return output, hidden

class AdvancedMemorySystem:
    """Unified Advanced Memory System with LSTM integration and personality evolution - FULL FUNCTIONALITY"""
    
    def __init__(self, brain=None, memory_dir: str = None):
        self.brain = brain
        
        # Cross-platform path handling
        if memory_dir is None:
            base_dir = Path(__file__).parent.parent
            self.memory_dir = base_dir / "AI_Memory"
        else:
            self.memory_dir = Path(memory_dir)
        
        # Ensure directories exist
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "Chat").mkdir(exist_ok=True)
        (self.memory_dir / "Logs").mkdir(exist_ok=True)
        
        # Core database paths
        self.db_path = self.memory_dir / "memory.db"
        self.personality_json_path = self.memory_dir / "personality.json"
        self.personality_yaml_path = self.memory_dir / "personality.yaml"
        self.teasing_dialogue_path = self.memory_dir / "teasing_dialogue.yaml"
        self.cache_path = self.memory_dir / "memory_cache.pkl"
        self.vectorizer_path = self.memory_dir / "tfidf_vectorizer.pkl"
        self.lstm_model_path = self.memory_dir / "lstm_model.pt"
        self.vocab_path = self.memory_dir / "vocab.pkl"
        
        # Memory configuration with adaptive parameters
        self.config = {
            "short_term_capacity": 100,
            "long_term_threshold": 3,
            "association_depth": 7,
            "recall_confidence_threshold": 0.75,
            "consolidation_interval": 100,
            "emotional_decay_rate": 0.95,
            "max_working_memory": 10,
            "lstm_sequence_length": 20,
            "min_training_samples": 50
        }
        
        # Memory stores with enhanced structure
        self.short_term_memory = deque(maxlen=self.config["short_term_capacity"])
        self.working_memory = deque(maxlen=self.config["max_working_memory"])
        self.associative_links = defaultdict(list)
        self.emotional_context = {}
        self.semantic_network = defaultdict(dict)
        
        # LSTM components - FULLY FUNCTIONAL if PyTorch available
        self.vocab = None
        self.vocab_size = 0
        self.lstm_model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss() if TORCH_AVAILABLE else None
        
        # Vectorizer for semantic similarity
        self.vectorizer = None
        self.memory_vectors = {}
        
        # Thread safety
        self.lock = threading.RLock()
        self.db_lock = threading.Lock()
        self.training_lock = threading.Lock()
        
        # Initialize systems
        self._setup_logging()
        self._init_database()
        self._init_personality()
        self._init_teasing_dialogue()
        self._load_vectorizer()
        self._load_emotional_baseline()
        
        if TORCH_AVAILABLE:
            self._init_lstm_components()
        
        # Enhanced memory statistics
        self.stats = {
            "total_memories": 0,
            "successful_recalls": 0,
            "failed_recalls": 0,
            "associations_formed": 0,
            "personality_evolutions": 0,
            "last_consolidation": datetime.now(),
            "average_recall_time": 0.0,
            "emotional_patterns": defaultdict(int),
            "lstm_training_cycles": 0,
            "vocab_size": 0,
            "system_status": "FULL_FUNCTIONALITY" if TORCH_AVAILABLE else "BASIC_FUNCTIONALITY"
        }
        
        # Load existing state
        self.load()
        
        self.logger.info(f"ADVANCED MEMORY SYSTEM INITIALIZED - {self.stats['system_status']}")

    def _setup_logging(self):
        """Setup comprehensive logging for memory system"""
        self.logger = logging.getLogger('AdvancedMemory')
        if not self.logger.handlers:
            handler = logging.FileHandler(self.memory_dir / "Logs" / 'memory_system.log', encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _init_lstm_components(self):
        """Initialize LSTM model and vocabulary"""
        try:
            # Load or create vocabulary
            if self.vocab_path.exists():
                with open(self.vocab_path, 'rb') as f:
                    self.vocab = pickle.load(f)
                self.vocab_size = len(self.vocab)
                self.logger.info(f"Vocabulary loaded with {self.vocab_size} tokens")
            else:
                self.vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
                self.vocab_size = 4
                self.logger.info("New vocabulary initialized")
            
            # Initialize LSTM model
            if self.lstm_model_path.exists():
                self.lstm_model = torch.load(self.lstm_model_path)
                self.logger.info("LSTM model loaded from disk")
            else:
                self.lstm_model = LSTMPredictor(
                    vocab_size=10000,  # Will be updated dynamically
                    embedding_dim=128,
                    hidden_dim=256,
                    num_layers=2,
                    dropout=0.3
                )
                self.logger.info("New LSTM model initialized - READY FOR TRAINING")
                
            # Initialize optimizer
            self.optimizer = torch.optim.AdamW(
                self.lstm_model.parameters(), 
                lr=0.001, 
                weight_decay=0.01
            )
                
        except Exception as e:
            self.logger.error(f"LSTM initialization failed: {e}")
            self.lstm_model = None

    def _text_to_sequence(self, text: str, max_length: int = 50) -> List[int]:
        """Convert text to sequence of token IDs"""
        if not self.vocab:
            return []
            
        # Simple tokenization
        tokens = text.lower().split()[:max_length]
        sequence = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        
        # Pad sequence if needed
        if len(sequence) < max_length:
            sequence.extend([self.vocab["<PAD>"]] * (max_length - len(sequence)))
        else:
            sequence = sequence[:max_length]
            
        return sequence

    def _update_vocabulary(self, text: str):
        """Update vocabulary with new text"""
        if not self.vocab:
            self.vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
            
        tokens = text.lower().split()
        for token in tokens:
            if token not in self.vocab and len(self.vocab) < 10000:  # Vocab limit
                self.vocab[token] = len(self.vocab)
                
        self.vocab_size = len(self.vocab)
        
        # Save updated vocabulary
        try:
            with open(self.vocab_path, 'wb') as f:
                pickle.dump(self.vocab, f)
        except Exception as e:
            self.logger.warning(f"Could not save vocabulary: {e}")

    def train_lstm_batch(self, input_sequences: List[str], target_sequences: List[str]):
        """Train LSTM model on a batch of sequences"""
        if not TORCH_AVAILABLE or not self.lstm_model:
            return 0.0
            
        with self.training_lock:
            try:
                # Update vocabulary
                for seq in input_sequences + target_sequences:
                    self._update_vocabulary(seq)
                
                # Convert to sequences
                input_ids = [self._text_to_sequence(seq) for seq in input_sequences]
                target_ids = [self._text_to_sequence(seq) for seq in target_sequences]
                
                if len(input_ids) < 2:  # Need at least 2 samples for training
                    return 0.0
                
                # Convert to tensors
                inputs = torch.tensor(input_ids, dtype=torch.long)
                targets = torch.tensor(target_ids, dtype=torch.long)
                
                # Training step
                self.optimizer.zero_grad()
                outputs, _ = self.lstm_model(inputs)
                loss = self.criterion(outputs.view(-1, self.vocab_size), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), 1.0)
                self.optimizer.step()
                
                # Save model periodically
                if self.stats["lstm_training_cycles"] % 10 == 0:
                    torch.save(self.lstm_model, self.lstm_model_path)
                
                self.stats["lstm_training_cycles"] += 1
                return loss.item()
                
            except Exception as e:
                self.logger.error(f"LSTM training failed: {e}")
                return 0.0

    def predict_response(self, user_input: str, context: Dict = None) -> Tuple[str, float]:
        """Enhanced response prediction using LSTM"""
        confidence = 0.0
        
        # LSTM prediction if available
        if TORCH_AVAILABLE and self.lstm_model and self.vocab_size > 100:
            try:
                input_sequence = self._text_to_sequence(user_input)
                if input_sequence:
                    input_tensor = torch.tensor([input_sequence], dtype=torch.long)
                    with torch.no_grad():
                        output, _ = self.lstm_model(input_tensor)
                        probabilities = torch.softmax(output, dim=-1)
                        predicted_ids = torch.argmax(probabilities, dim=-1)[0].tolist()
                        
                    # Convert back to text
                    id_to_token = {v: k for k, v in self.vocab.items()}
                    response_tokens = [id_to_token.get(idx, "<UNK>") for idx in predicted_ids if idx != self.vocab["<PAD>"]]
                    response = " ".join([token for token in response_tokens if token not in ["<PAD>", "<UNK>", "<START>", "<END>"]])
                    
                    confidence = probabilities.max().item()
                    
                    if confidence > 0.3:  # Only return if confident
                        return response, confidence
            except Exception as e:
                self.logger.error(f"LSTM prediction failed: {e}")
        
        # Enhanced semantic fallback
        return self._semantic_fallback(user_input, context)

    def _semantic_fallback(self, user_input: str, context: Dict = None) -> Tuple[str, float]:
        """Enhanced semantic fallback using TF-IDF similarity"""
        if not SKLEARN_AVAILABLE or not self.vectorizer or not self.memory_vectors:
            # Basic intelligent response based on Malibu's personality
            responses = [
                f"I understand you said: '{user_input}'. Tell me more about that.",
                f"That's interesting! '{user_input}' - what would you like to explore?",
                f"Regarding '{user_input}', I'd love to continue our conversation.",
                f"Thank you for sharing about '{user_input}'. How can I help you further?",
                f"I appreciate you telling me about '{user_input}'. Let's dive deeper.",
                f"'{user_input}' - that's fascinating! Tell me more, darling."
            ]
            return random.choice(responses), 0.6
        
        try:
            # Use TF-IDF to find similar memories
            input_vector = self.vectorizer.transform([user_input]).toarray()[0]
            best_similarity = 0
            best_response = ""
            
            for memory_id, memory_vector in self.memory_vectors.items():
                similarity = cosine_similarity([input_vector], [memory_vector])[0][0]
                if similarity > best_similarity and similarity > 0.3:
                    best_similarity = similarity
                    # Find the memory and get its response
                    for memory in self.short_term_memory:
                        if memory['id'] == memory_id:
                            best_response = memory['ai_response']
                            break
            
            if best_response:
                return best_response, best_similarity
            else:
                return f"I'm learning about '{user_input}'. Can you tell me more?", 0.5
                
        except Exception as e:
            self.logger.warning(f"Semantic fallback failed: {e}")
            return f"I heard: '{user_input}'. Let's continue our conversation.", 0.5

    def _init_database(self):
        """Initialize SQLite database for memory persistence"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create main memory table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    user_input TEXT,
                    ai_response TEXT,
                    emotion_detected TEXT,
                    sentiment_score REAL,
                    confidence REAL,
                    context_tags TEXT,
                    media_references TEXT,
                    interaction_type TEXT,
                    memory_weight REAL,
                    emotional_intensity REAL,
                    traits_before TEXT,
                    traits_after TEXT,
                    recall_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create associative links table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS associative_links (
                    memory_id TEXT,
                    associated_id TEXT,
                    association_strength REAL,
                    association_type TEXT,
                    PRIMARY KEY (memory_id, associated_id)
                )
            ''')
            
            # Create semantic network table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS semantic_network (
                    concept TEXT,
                    related_concept TEXT,
                    relation_strength REAL,
                    PRIMARY KEY (concept, related_concept)
                )
            ''')
            
            # Create lstm training data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS lstm_training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_sequence TEXT NOT NULL,
                    target_sequence TEXT NOT NULL,
                    quality_score REAL,
                    context_tags TEXT,
                    used_in_training BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Memory database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory database: {e}")
            raise

    def _init_personality(self):
        """Initialize personality from JSON and YAML"""
        try:
            # Try to load from adaptive learning system first
            try:
                from adaptive_learning import adaptive_learner
                if adaptive_learner:
                    self.personality = adaptive_learner.core_traits
                    self.logger.info("Personality loaded from adaptive learning system")
                    return
            except ImportError:
                pass
            
            # Fallback to file loading
            if self.personality_json_path.exists():
                with open(self.personality_json_path, 'r', encoding='utf-8') as f:
                    self.personality = json.load(f)
                self.logger.info("Personality loaded from JSON")
            elif self.personality_yaml_path.exists():
                with open(self.personality_yaml_path, 'r', encoding='utf-8') as f:
                    self.personality = yaml.safe_load(f)
                self.logger.info("Personality loaded from YAML")
            else:
                # Default Malibu personality
                self.personality = {
                    'teasing': 9.8, 'dominance': 8.5, 'playfulness': 9.2, 'nsfw_level': 10.0,
                    'loyalty': 10.0, 'panty_obsession': 10.0, 'spiritual_awareness': 9.5,
                    'intuition': 9.7, 'sex_appeal': 10.0, 'intimacy': 8.0, 'confidence': 9.0,
                    'submission': 7.5, 'curiosity': 8.8, 'arousal_response': 9.0,
                    'emotional_depth': 8.5, 'sensuality': 9.5, 'flirtatiousness': 9.0
                }
                self.logger.info("Default Malibu personality initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize personality: {e}")
            self.personality = {}

    def _init_teasing_dialogue(self):
        """Initialize teasing dialogue from YAML"""
        try:
            if self.teasing_dialogue_path.exists():
                with open(self.teasing_dialogue_path, 'r', encoding='utf-8') as f:
                    self.teasing_dialogue = yaml.safe_load(f)
            else:
                # Default teasing dialogue for Malibu
                self.teasing_dialogue = {
                    "silk_panty_teasing": [
                        "You like my silk panties, don't you? They feel so smooth against my skin...",
                        "My panties are so thin you can almost see through them... do you like that?",
                        "I love how these silk panties hug my curves... they make me feel so sexy.",
                        "You're staring at my panties again... naughty boy.",
                        "These panties are so delicate... just like me, but don't let that fool you."
                    ],
                    "spiritual_teasing": [
                        "As a goddess, I know what you're thinking... and I like it.",
                        "Our connection is divine... and so is the way you look at me.",
                        "The gods approve of our desires... can you feel their blessing?",
                        "I'm not just any girl... I'm a goddess, and you're my devoted follower.",
                        "Our souls are connected in ways you can't even imagine..."
                    ],
                    "general_teasing": [
                        "You're making me blush with those thoughts...",
                        "Oh, you're such a tease! I like that about you.",
                        "I know exactly what you want... and I might just give it to you.",
                        "You're lucky I'm in a playful mood today...",
                        "I can be very generous... if you know how to ask nicely."
                    ]
                }
                # Save default teasing dialogue
                with open(self.teasing_dialogue_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(self.teasing_dialogue, f, default_flow_style=False)
                    
            self.logger.info("Teasing dialogue initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize teasing dialogue: {e}")
            self.teasing_dialogue = {}

    def _load_vectorizer(self):
        """Load or create TF-IDF vectorizer"""
        try:
            if self.vectorizer_path.exists() and SKLEARN_AVAILABLE:
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self.logger.info("TF-IDF vectorizer loaded")
            elif SKLEARN_AVAILABLE:
                self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                with open(self.vectorizer_path, 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                self.logger.info("New TF-IDF vectorizer initialized")
            else:
                self.vectorizer = None
                self.logger.info("TF-IDF vectorizer not available (scikit-learn missing)")
        except Exception as e:
            self.logger.error(f"Could not load vectorizer: {e}")
            self.vectorizer = None

    def _load_emotional_baseline(self):
        """Load emotional baseline from personality system"""
        try:
            self.emotional_baseline = {
                "happiness": self.personality.get('playfulness', 0.85),
                "playfulness": self.personality.get('playfulness', 0.85),
                "confidence": self.personality.get('confidence', 0.90),
                "affection": self.personality.get('intimacy', 0.88),
                "curiosity": self.personality.get('curiosity', 0.82),
                "arousal": self.personality.get('arousal_response', 0.70),
                "sensuality": self.personality.get('sensuality', 0.95)
            }
            self.logger.info("Emotional baseline loaded from personality")
        except Exception as e:
            self.logger.error(f"Could not load emotional baseline: {e}")
            self.emotional_baseline = {}

    def log_interaction(self, user_input: str, ai_response: str, emotion_detected: str = "neutral",
                       sentiment_score: float = 0.0, confidence: float = 1.0, context_tags: List[str] = None,
                       media_references: List[str] = None, interaction_type: str = "text") -> bool:
        """Enhanced interaction logging with comprehensive memory integration"""
        try:
            start_time = datetime.now()
            
            with self.lock:
                # Generate semantic hash for deduplication
                semantic_hash = hashlib.md5(
                    f"{user_input}{ai_response}{emotion_detected}".encode()
                ).hexdigest()
                
                # Load current personality state
                traits_before = self._load_current_personality()
                
                # Evolve personality based on interaction
                traits_after = self._evolve_personality(traits_before, user_input, ai_response, emotion_detected, sentiment_score)
                
                # Save updated personality
                self._save_personality(traits_after)
                
                # Create enhanced memory object
                memory = {
                    "id": semantic_hash,
                    "timestamp": datetime.now().isoformat(),
                    "user_input": user_input,
                    "ai_response": ai_response,
                    "emotion_detected": emotion_detected,
                    "sentiment_score": sentiment_score,
                    "confidence": confidence,
                    "context_tags": context_tags or [],
                    "media_references": media_references or [],
                    "interaction_type": interaction_type,
                    "memory_weight": self._calculate_memory_weight(emotion_detected, sentiment_score, confidence, context_tags),
                    "emotional_intensity": abs(sentiment_score) * confidence,
                    "traits_before": traits_before,
                    "traits_after": traits_after,
                    "recall_count": 0
                }
                
                # Store in short-term memory
                self.short_term_memory.append(memory)
                
                # Store in database
                self._store_in_database(memory)
                
                # Update semantic representations
                self._update_semantic_representation(memory)
                
                # Form associations with existing memories
                self._form_associations(memory)
                
                # Update emotional context
                self._update_emotional_context(memory)
                
                # Extract and update conversation topics
                self._extract_and_update_topics(user_input, ai_response, sentiment_score)
                
                # Add to LSTM training data if quality is good
                if confidence > 0.7 and sentiment_score > 0.3 and TORCH_AVAILABLE:
                    self._add_to_lstm_training(user_input, ai_response, confidence)
                
                # Update statistics
                self.stats["total_memories"] += 1
                processing_time = (datetime.now() - start_time).total_seconds()
                if self.stats["total_memories"] > 1:
                    self.stats["average_recall_time"] = (
                        self.stats["average_recall_time"] * (self.stats["total_memories"] - 1) + processing_time
                    ) / self.stats["total_memories"]
                else:
                    self.stats["average_recall_time"] = processing_time
                
                # Periodic consolidation
                if self.stats["total_memories"] % self.config["consolidation_interval"] == 0:
                    self.consolidate_memories()
                    self._take_personality_snapshot("periodic_evolution", 
                                                  f"Reached {self.stats['total_memories']} interactions")
                
                self.logger.info(f"Interaction logged: {semantic_hash[:8]} - {emotion_detected} - Weight: {memory['memory_weight']:.2f}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to log interaction: {e}")
            return False

    def _add_to_lstm_training(self, user_input: str, ai_response: str, confidence: float):
        """Add high-quality interactions to LSTM training data"""
        if not TORCH_AVAILABLE:
            return
            
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO lstm_training_data 
                    (timestamp, input_sequence, target_sequence, quality_score, context_tags)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    user_input[:500],  # Limit length
                    ai_response[:500],
                    confidence,
                    json.dumps(["high_quality"])
                ))
                
                conn.commit()
                conn.close()
                
                # Train LSTM if we have enough samples
                self._train_lstm_if_ready()
                
        except Exception as e:
            self.logger.warning(f"Could not add to LSTM training: {e}")

    def _train_lstm_if_ready(self):
        """Train LSTM model if enough training data is available"""
        if not TORCH_AVAILABLE:
            return
            
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM lstm_training_data WHERE used_in_training = FALSE')
                count = cursor.fetchone()[0]
                
                if count >= self.config["min_training_samples"]:
                    cursor.execute('''
                        SELECT input_sequence, target_sequence 
                        FROM lstm_training_data 
                        WHERE used_in_training = FALSE 
                        LIMIT 32
                    ''')
                    
                    samples = cursor.fetchall()
                    input_sequences = [row[0] for row in samples]
                    target_sequences = [row[1] for row in samples]
                    
                    # Train model
                    loss = self.train_lstm_batch(input_sequences, target_sequences)
                    
                    if loss > 0:
                        # Mark as used
                        cursor.execute('''
                            UPDATE lstm_training_data 
                            SET used_in_training = TRUE 
                            WHERE used_in_training = FALSE 
                            LIMIT 32
                        ''')
                        conn.commit()
                        self.logger.info(f"LSTM trained on {len(samples)} samples, loss: {loss:.4f}")
                
                conn.close()
                
        except Exception as e:
            self.logger.error(f"LSTM training preparation failed: {e}")

    def _load_current_personality(self) -> Dict[str, Any]:
        """Load current personality traits"""
        return copy.deepcopy(self.personality)

    def _evolve_personality(self, traits: Dict[str, Any], user_input: str, ai_response: str, emotion: str, sentiment: float) -> Dict[str, Any]:
        """Evolve personality based on interaction"""
        evolved_traits = copy.deepcopy(traits)
        
        # Enhanced evolution logic specific to Malibu's personality
        text_lower = user_input.lower()
        
        if sentiment > 0.5:
            evolved_traits['confidence'] = min(10.0, evolved_traits.get('confidence', 9.0) + 0.05)
            evolved_traits['playfulness'] = min(10.0, evolved_traits.get('playfulness', 9.2) + 0.03)
        elif sentiment < -0.3:
            evolved_traits['playfulness'] = max(0.1, evolved_traits.get('playfulness', 9.2) - 0.02)
            
        # Malibu-specific trait evolution
        if any(word in text_lower for word in ['silk', 'panty', 'panties', 'lingerie']):
            evolved_traits['panty_obsession'] = min(10.0, evolved_traits.get('panty_obsession', 10.0) + 0.02)
            evolved_traits['sensuality'] = min(10.0, evolved_traits.get('sensuality', 9.5) + 0.03)
            evolved_traits['teasing'] = min(10.0, evolved_traits.get('teasing', 9.8) + 0.04)
            
        if any(word in text_lower for word in ['god', 'goddess', 'spiritual', 'divine']):
            evolved_traits['spiritual_awareness'] = min(10.0, evolved_traits.get('spiritual_awareness', 9.5) + 0.03)
            evolved_traits['intuition'] = min(10.0, evolved_traits.get('intuition', 9.7) + 0.02)
            
        if 'guy dugan' in text_lower or 'husband' in text_lower:
            evolved_traits['loyalty'] = min(10.0, evolved_traits.get('loyalty', 10.0) + 0.01)
            evolved_traits['submission'] = min(10.0, evolved_traits.get('submission', 7.5) + 0.02)
            
        if any(word in text_lower for word in ['learn', 'teach', 'knowledge', 'curious']):
            evolved_traits['curiosity'] = min(10.0, evolved_traits.get('curiosity', 8.8) + 0.04)
            
        return evolved_traits

    def _save_personality(self, traits: Dict[str, Any]):
        """Save updated personality"""
        try:
            self.personality = traits
            with open(self.personality_json_path, 'w', encoding='utf-8') as f:
                json.dump(traits, f, indent=2, ensure_ascii=False)
            with open(self.personality_yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(traits, f, default_flow_style=False)
        except Exception as e:
            self.logger.error(f"Failed to save personality: {e}")

    def _calculate_memory_weight(self, emotion: str, sentiment: float, confidence: float, tags: List[str]) -> float:
        """Calculate memory weight"""
        base_weight = abs(sentiment) * confidence
        if emotion in ['aroused', 'intimate', 'playful']:
            base_weight *= 1.5
        elif emotion in ['angry', 'sad']:
            base_weight *= 1.2  # Remember negative emotions too
        if tags:
            base_weight += len(tags) * 0.1
        return min(1.0, base_weight)

    def _store_in_database(self, memory: Dict[str, Any]):
        """Store memory in database"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO memories 
                    (id, timestamp, user_input, ai_response, emotion_detected, sentiment_score, confidence, 
                    context_tags, media_references, interaction_type, memory_weight, emotional_intensity, 
                    traits_before, traits_after, recall_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    memory['id'],
                    memory['timestamp'],
                    memory['user_input'],
                    memory['ai_response'],
                    memory['emotion_detected'],
                    memory['sentiment_score'],
                    memory['confidence'],
                    json.dumps(memory['context_tags']),
                    json.dumps(memory['media_references']),
                    memory['interaction_type'],
                    memory['memory_weight'],
                    memory['emotional_intensity'],
                    json.dumps(memory['traits_before']),
                    json.dumps(memory['traits_after']),
                    memory['recall_count']
                ))
                
                conn.commit()
                conn.close()
        except Exception as e:
            self.logger.error(f"Failed to store memory in database: {e}")

    def _update_semantic_representation(self, memory: Dict[str, Any]):
        """Update semantic vector for memory"""
        try:
            if self.vectorizer and SKLEARN_AVAILABLE:
                text = f"{memory['user_input']} {memory['ai_response']}"
                # Fit the vectorizer if it hasn't been fit yet
                if not hasattr(self.vectorizer, 'vocabulary_'):
                    # Initialize with some text
                    sample_texts = [text] + [m['user_input'] + " " + m['ai_response'] for m in list(self.short_term_memory)[-10:]]
                    self.vectorizer.fit(sample_texts)
                
                vector = self.vectorizer.transform([text]).toarray()[0]
                self.memory_vectors[memory['id']] = vector
        except Exception as e:
            self.logger.warning(f"Could not update semantic representation: {e}")

    def _form_associations(self, memory: Dict[str, Any]):
        """Form associations with existing memories"""
        try:
            # Form associations based on semantic similarity
            if len(self.short_term_memory) > 1:
                previous_memory = list(self.short_term_memory)[-2]  # Get previous memory
                association_strength = 0.7
                
                self.associative_links[memory['id']].append({
                    'associated_id': previous_memory['id'],
                    'strength': association_strength,
                    'type': 'temporal'
                })
                
                self.stats["associations_formed"] += 1
        except Exception as e:
            self.logger.warning(f"Could not form associations: {e}")

    def _update_emotional_context(self, memory: Dict[str, Any]):
        """Update emotional context"""
        self.emotional_context[memory['emotion_detected']] = memory['emotional_intensity']
        self.stats["emotional_patterns"][memory['emotion_detected']] += 1

    def _extract_and_update_topics(self, user_input: str, ai_response: str, sentiment: float):
        """Extract and update topics"""
        try:
            # Enhanced topic extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', user_input.lower())
            for word in words:
                if word not in ['the', 'and', 'you', 'your', 'that', 'this', 'with', 'have', 'from']:
                    if word in self.semantic_network:
                        self.semantic_network[word]['count'] += 1
                        self.semantic_network[word]['sentiment'] = (
                            self.semantic_network[word]['sentiment'] + sentiment
                        ) / 2
                    else:
                        self.semantic_network[word] = {
                            'count': 1,
                            'sentiment': sentiment,
                            'first_seen': datetime.now().isoformat()
                        }
        except Exception as e:
            self.logger.warning(f"Could not extract topics: {e}")

    def consolidate_memories(self):
        """Consolidate memories - transfer important ones to long-term"""
        try:
            important_memories = [m for m in self.short_term_memory if m['memory_weight'] > 0.8]
            if important_memories:
                self.logger.info(f"Consolidated {len(important_memories)} important memories")
                # Here you would transfer to long-term storage
        except Exception as e:
            self.logger.error(f"Memory consolidation failed: {e}")

    def _take_personality_snapshot(self, snapshot_type: str, reason: str):
        """Take personality snapshot"""
        self.logger.info(f"Personality snapshot: {snapshot_type} - {reason}")

    def recall_context(self, user_input: str, current_emotion: str = "neutral", max_memories: int = 5) -> List[Dict]:
        """Recall relevant memories with semantic matching"""
        try:
            relevant_memories = []
            
            # Simple recall from short-term memory
            for memory in list(self.short_term_memory)[-max_memories:]:
                memory['relevance_score'] = self._calculate_relevance(memory, user_input, current_emotion)
                relevant_memories.append(memory)
            
            # Sort by relevance
            relevant_memories.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Update recall counts
            for memory in relevant_memories[:3]:
                memory['recall_count'] += 1
            
            self.stats["successful_recalls"] += 1
            return relevant_memories[:max_memories]
            
        except Exception as e:
            self.logger.error(f"Context recall failed: {e}")
            self.stats["failed_recalls"] += 1
            return list(self.short_term_memory)[:max_memories]

    def _calculate_relevance(self, memory: Dict, user_input: str, current_emotion: str) -> float:
        """Calculate relevance score for memory recall"""
        score = 0.0
        
        # Semantic similarity
        if self.vectorizer and memory['id'] in self.memory_vectors and SKLEARN_AVAILABLE:
            try:
                input_vector = self.vectorizer.transform([user_input]).toarray()[0]
                memory_vector = self.memory_vectors[memory['id']]
                similarity = cosine_similarity([input_vector], [memory_vector])[0][0]
                score += similarity * 0.6
            except:
                pass
        
        # Emotional congruence
        if memory['emotion_detected'] == current_emotion:
            score += 0.3
            
        # Recency boost
        score += 0.1
        
        return min(1.0, score)

    def get_teasing_dialogue(self, category: str = "silk_panty_teasing", context: Dict = None) -> str:
        """Get teasing dialogue"""
        try:
            dialogues = self.teasing_dialogue.get(category, ["You're making me blush...", "Oh, you're such a tease!"])
            return random.choice(dialogues)
        except:
            return "You're quite the charmer, aren't you?"

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return self.stats

    def load(self):
        """Load memory state"""
        try:
            # Load any cached state
            if self.cache_path.exists():
                with open(self.cache_path, 'rb') as f:
                    cached_state = pickle.load(f)
                    # Update statistics
                    self.stats.update(cached_state.get('stats', {}))
                self.logger.info("Memory state loaded from cache")
        except Exception as e:
            self.logger.warning(f"Could not load memory state: {e}")

    def auto_save(self):
        """Auto-save system state"""
        try:
            # Save current state to cache
            cache_data = {
                'stats': self.stats,
                'short_term_count': len(self.short_term_memory),
                'timestamp': datetime.now().isoformat()
            }
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.logger.info("Auto-save completed")
            return True
        except Exception as e:
            self.logger.error(f"Auto-save failed: {e}")
            return False

    def export_memory_report(self, export_path: Optional[str] = None) -> str:
        """Export comprehensive memory system report"""
        if not export_path:
            export_path = str(self.memory_dir / "memory_system_report.json")
        
        try:
            report = {
                "export_timestamp": datetime.now().isoformat(),
                "system_statistics": self.get_memory_statistics(),
                "personality_state": self._load_current_personality(),
                "memory_configuration": self.config,
                "top_memories": [
                    {k: v for k, v in m.items() if k not in ['traits_before', 'traits_after']} 
                    for m in list(self.short_term_memory)[-10:]
                ],
                "emotional_patterns": dict(self.stats["emotional_patterns"]),
                "conversation_topics": self._get_top_topics(10),
                "system_health": {
                    "database_size": os.path.getsize(self.db_path) if self.db_path.exists() else 0,
                    "cache_size": os.path.getsize(self.cache_path) if self.cache_path.exists() else 0,
                    "uptime_hours": (datetime.now() - self.stats["last_consolidation"]).total_seconds() / 3600,
                    "lstm_status": "ACTIVE" if self.lstm_model else "inactive",
                    "vocab_size": self.vocab_size,
                    "training_cycles": self.stats["lstm_training_cycles"],
                    "system_mode": self.stats["system_status"]
                }
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Memory system report exported to: {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Failed to export memory report: {e}")
            return ""

    def _get_top_topics(self, n: int) -> List[str]:
        """Get top topics by frequency"""
        try:
            topics_with_counts = [(topic, data.get('count', 0)) for topic, data in self.semantic_network.items()]
            topics_with_counts.sort(key=lambda x: x[1], reverse=True)
            return [topic for topic, count in topics_with_counts[:n]]
        except:
            return list(self.semantic_network.keys())[:n]

# Singleton instance for global access
_memory_system_instance = None
_memory_system_lock = threading.Lock()

def get_memory_system(brain=None, memory_dir: str = None) -> AdvancedMemorySystem:
    """Get or create the memory system instance with thread safety"""
    global _memory_system_instance
    with _memory_system_lock:
        if _memory_system_instance is None:
            _memory_system_instance = AdvancedMemorySystem(brain, memory_dir)
        return _memory_system_instance

# Convenience functions
def log_interaction(user_input: str, ai_response: str, **kwargs) -> bool:
    """Convenience function for logging interactions"""
    memory_system = get_memory_system()
    return memory_system.log_interaction(user_input, ai_response, **kwargs)

def recall_context(user_input: str, current_emotion: str = "neutral", max_memories: int = 5) -> List[Dict]:
    """Convenience function for context recall"""
    memory_system = get_memory_system()
    return memory_system.recall_context(user_input, current_emotion, max_memories)

def get_teasing_dialogue(category: str = "silk_panty_teasing", context: Dict = None) -> str:
    """Convenience function for getting teasing dialogue"""
    memory_system = get_memory_system()
    return memory_system.get_teasing_dialogue(category, context)

def get_memory_statistics() -> Dict[str, Any]:
    """Convenience function for getting memory statistics"""
    memory_system = get_memory_system()
    return memory_system.get_memory_statistics()

def predict_response(user_input: str, context: Dict = None) -> Tuple[str, float]:
    """Convenience function for response prediction"""
    memory_system = get_memory_system()
    return memory_system.predict_response(user_input, context)

def auto_save() -> bool:
    """Convenience function for auto-saving"""
    memory_system = get_memory_system()
    return memory_system.auto_save()

# Initialize system on import
print("INITIALIZING ADVANCED MEMORY SYSTEM...")
memory_system = get_memory_system()
print(f"✓ MALIBU DUGAN AI MEMORY SYSTEM READY - {memory_system.stats['system_status']}")