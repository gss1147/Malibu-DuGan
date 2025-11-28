"""
Quick Fix Script for Malibu DuGan AI System Errors
"""

import sqlite3
import os
import logging

def create_missing_tables():
    """Create missing database tables"""
    db_path = r"X:\Malibu_DuGan\AI_Memory\malibu_personality.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create emotional_history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotional_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                emotion TEXT NOT NULL,
                intensity REAL NOT NULL,
                trigger_text TEXT,
                context TEXT
            )
        ''')
        
        # Create personality_evolution table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personality_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trait TEXT NOT NULL,
                old_value REAL NOT NULL,
                new_value REAL NOT NULL,
                trigger_type TEXT,
                context TEXT
            )
        ''')
        
        # Create trait_evolutions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trait_evolutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trait TEXT NOT NULL,
                old_value REAL NOT NULL,
                new_value REAL NOT NULL,
                trigger_type TEXT,
                context TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✓ Missing database tables created successfully")
        
    except Exception as e:
        print(f"✗ Error creating tables: {e}")

def fix_ar_system():
    """Provide AR system fixes"""
    print("AR System Fixes:")
    print("1. Ensure camera is not being used by other applications")
    print("2. Try different camera indexes (0, 1, 2)")
    print("3. Check camera permissions")
    print("4. Verify OpenCV can access camera")

def check_dependencies():
    """Check for common dependency issues"""
    issues = []
    
    try:
        import torch
        print("✓ PyTorch: OK")
    except ImportError:
        issues.append("PyTorch not installed")
    
    try:
        import cv2
        print("✓ OpenCV: OK")
    except ImportError:
        issues.append("OpenCV not installed")
    
    try:
        import pygame
        print("✓ Pygame: OK")
    except ImportError:
        issues.append("Pygame not installed")
    
    try:
        import mediapipe
        print("✓ MediaPipe: OK")
    except ImportError:
        issues.append("MediaPipe not installed")
    
    return issues

if __name__ == "__main__":
    print("Running Malibu DuGan System Fixes...")
    
    # Create missing tables
    create_missing_tables()
    
    # Check dependencies
    print("\nDependency Check:")
    issues = check_dependencies()
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ All dependencies OK")
    
    # AR fixes
    print("\n" + "="*50)
    fix_ar_system()
    
    print("\nFix script completed! Try running your tests again.")