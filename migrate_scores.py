
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "storage" / "conversations.db"

def migrate_scores():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return

    print(f"Migrating scores in {DB_PATH}...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if we need to migrate by looking for small scores
        cursor.execute("SELECT count(*) FROM model_metrics WHERE score_overall <= 1.0 AND score_overall > 0")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("No scores <= 1.0 found. Migration might not be needed.")
        else:
            print(f"Found {count} records with scores <= 1.0. Updating to 0-100 scale...")
            
            # Update all score columns
            columns = [
                "score_overall", 
                "score_coherence", 
                "score_reasoning", 
                "score_engagement", 
                "score_instruction_following", 
                "score_tool_usage"
            ]
            
            for col in columns:
                # Multiply by 100 where score is <= 1.0 and not null
                query = f"UPDATE model_metrics SET {col} = {col} * 100 WHERE {col} <= 1.0 AND {col} IS NOT NULL"
                cursor.execute(query)
                
            conn.commit()
            print("Migration completed successfully.")
            
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_scores()
