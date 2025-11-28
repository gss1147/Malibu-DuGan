-- Fix for personality_state table
CREATE TABLE IF NOT EXISTS personality_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    emotional_state TEXT,
    current_mood TEXT,
    personality_traits TEXT,
    interests TEXT,
    conversation_context TEXT
);

-- Fix for emotional_history table  
CREATE TABLE IF NOT EXISTS emotional_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    emotion_type TEXT NOT NULL,
    intensity REAL DEFAULT 0.5,
    trigger_context TEXT,
    duration_seconds INTEGER
);

-- Fix for memory_log table
CREATE TABLE IF NOT EXISTS memory_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    weight REAL DEFAULT 0.5,
    semantic_representation TEXT
);

-- Fix for lstm_training_data table
CREATE TABLE IF NOT EXISTS lstm_training_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    input_sequence TEXT NOT NULL,
    target_sequence TEXT NOT NULL,
    sequence_length INTEGER
);