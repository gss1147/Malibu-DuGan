import sqlite3
import os
import logging

def initialize_database():
    """Initialize all database tables with proper schemas"""
    
    # Ensure directory exists
    base_dir = r"X:\Malibu_DuGan"
    db_path = os.path.join(base_dir, "AI_Memory", "memory.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create all tables with proper schemas
        cursor.executescript('''
            -- Personality state table
            CREATE TABLE IF NOT EXISTS personality_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                emotional_state TEXT,
                current_mood TEXT,
                personality_traits TEXT,
                interests TEXT,
                conversation_context TEXT
            );
            
            -- Emotional history table
            CREATE TABLE IF NOT EXISTS emotional_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                emotion_type TEXT NOT NULL,
                intensity REAL DEFAULT 0.5,
                trigger_context TEXT,
                duration_seconds INTEGER
            );
            
            -- Memory log table
            CREATE TABLE IF NOT EXISTS memory_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                weight REAL DEFAULT 0.5,
                semantic_representation TEXT
            );
            
            -- LSTM training data table
            CREATE TABLE IF NOT EXISTS lstm_training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                input_sequence TEXT NOT NULL,
                target_sequence TEXT NOT NULL,
                sequence_length INTEGER
            );
            
            -- Reasoning trees table (for deep reasoning)
            CREATE TABLE IF NOT EXISTS reasoning_trees (
                session_id TEXT PRIMARY KEY,
                tree_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                context TEXT NOT NULL
            );
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    initialize_database()