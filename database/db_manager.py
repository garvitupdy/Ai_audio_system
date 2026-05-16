import sqlite3
import json
from datetime import datetime
from pathlib import Path
import shutil

class TranscriptDatabase:
    def __init__(self, db_path="database/transcripts.db"):
        self.db_path = db_path
        self.vector_base_path = Path("database/vectors")
        self.vector_base_path.mkdir(parents=True, exist_ok=True)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                language TEXT NOT NULL,
                transcript TEXT NOT NULL,
                summary TEXT,
                action_items TEXT,
                key_decisions TEXT,
                open_questions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def add_transcript(self, data: dict):
        """Add new transcript and manage limit of 10"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
  
        cursor.execute("SELECT COUNT(*) FROM transcripts")
        count = cursor.fetchone()[0]
        
        if count >= 10:
           
            cursor.execute("SELECT id FROM transcripts ORDER BY created_at ASC LIMIT 1")
            oldest_id = cursor.fetchone()[0]
     
            cursor.execute("DELETE FROM transcripts WHERE id = ?", (oldest_id,))
            

            self._delete_vector_store(oldest_id)
        
        
        cursor.execute('''
            INSERT INTO transcripts 
            (title, source, language, transcript, summary, action_items, key_decisions, open_questions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('title', 'Untitled'),
            data.get('source', ''),
            data.get('language', 'english'),
            data.get('transcript', ''),
            data.get('summary', ''),
            json.dumps(data.get('action_items', [])) if isinstance(data.get('action_items'), list) else data.get('action_items', ''),
            json.dumps(data.get('key_decisions', [])) if isinstance(data.get('key_decisions'), list) else data.get('key_decisions', ''),
            json.dumps(data.get('open_questions', [])) if isinstance(data.get('open_questions'), list) else data.get('open_questions', '')
        ))
        
        transcript_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return transcript_id
    
    def _delete_vector_store(self, transcript_id: int):
        """Delete Chroma vector store for a transcript"""
        vector_dir = self.vector_base_path / f"chroma_{transcript_id}"
        if vector_dir.exists():
            shutil.rmtree(vector_dir)
            print(f"🗑️ Deleted vector store for transcript {transcript_id}")
    
    def get_all_transcripts(self):
        """Get all transcripts (newest first)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, title, source, language, created_at 
            FROM transcripts 
            ORDER BY created_at DESC
        ''')
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_transcript(self, transcript_id: int):
        """Get specific transcript"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM transcripts WHERE id = ?
        ''', (transcript_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'title': result[1],
                'source': result[2],
                'language': result[3],
                'transcript': result[4],
                'summary': result[5],
                'action_items': result[6],
                'key_decisions': result[7],
                'open_questions': result[8],
                'created_at': result[9]
            }
        return None
    
    def delete_transcript(self, transcript_id: int):
        """Delete specific transcript"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transcripts WHERE id = ?", (transcript_id,))
        conn.commit()
        conn.close()
        
        
        self._delete_vector_store(transcript_id)
    
    def update_analysis(self, transcript_id: int, field: str, value: str):
        """Update specific analysis field"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f'''
            UPDATE transcripts 
            SET {field} = ? 
            WHERE id = ?
        ''', (value, transcript_id))
        conn.commit()
        conn.close()