"""
Data Manager - SQLite database & CSV management for TinNam data.
Handles storage, retrieval, validation and deduplication.
"""
import sqlite3
import csv
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tinnam_data.db')
CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def get_db():
    """Get database connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_db()
    c = conn.cursor()
    
    # Mega 6/45 table
    c.execute('''CREATE TABLE IF NOT EXISTS mega645 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        draw_date TEXT NOT NULL,
        n1 INTEGER NOT NULL,
        n2 INTEGER NOT NULL,
        n3 INTEGER NOT NULL,
        n4 INTEGER NOT NULL,
        n5 INTEGER NOT NULL,
        n6 INTEGER NOT NULL,
        jackpot TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(draw_date, n1, n2, n3, n4, n5, n6)
    )''')
    
    # Power 6/55 table
    c.execute('''CREATE TABLE IF NOT EXISTS power655 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        draw_date TEXT NOT NULL,
        n1 INTEGER NOT NULL,
        n2 INTEGER NOT NULL,
        n3 INTEGER NOT NULL,
        n4 INTEGER NOT NULL,
        n5 INTEGER NOT NULL,
        n6 INTEGER NOT NULL,
        bonus INTEGER NOT NULL,
        jackpot TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(draw_date, n1, n2, n3, n4, n5, n6, bonus)
    )''')
    
    conn.commit()
    conn.close()
    print(f"[DB] Database initialized at {DB_PATH}")


def insert_mega645(rows):
    """Insert Mega 6/45 results. rows = list of (date, n1..n6, jackpot)."""
    conn = get_db()
    c = conn.cursor()
    inserted = 0
    for row in rows:
        try:
            c.execute('''INSERT OR IGNORE INTO mega645 
                        (draw_date, n1, n2, n3, n4, n5, n6, jackpot) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', row)
            if c.rowcount > 0:
                inserted += 1
        except Exception as e:
            print(f"[DB] Error inserting Mega row: {e}")
    conn.commit()
    conn.close()
    print(f"[DB] Mega 6/45: Inserted {inserted}/{len(rows)} rows")
    return inserted


def insert_power655(rows):
    """Insert Power 6/55 results. rows = list of (date, n1..n6, bonus, jackpot)."""
    conn = get_db()
    c = conn.cursor()
    inserted = 0
    for row in rows:
        try:
            c.execute('''INSERT OR IGNORE INTO power655 
                        (draw_date, n1, n2, n3, n4, n5, n6, bonus, jackpot) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', row)
            if c.rowcount > 0:
                inserted += 1
        except Exception as e:
            print(f"[DB] Error inserting Power row: {e}")
    conn.commit()
    conn.close()
    print(f"[DB] Power 6/55: Inserted {inserted}/{len(rows)} rows")
    return inserted


def get_mega645_all():
    """Get all Mega 6/45 results sorted by date."""
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM mega645 ORDER BY draw_date ASC'
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_power655_all():
    """Get all Power 6/55 results sorted by date."""
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM power655 ORDER BY draw_date ASC'
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_mega645_numbers():
    """Get Mega 6/45 numbers only as list of lists [[n1..n6], ...]."""
    rows = get_mega645_all()
    return [[r['n1'], r['n2'], r['n3'], r['n4'], r['n5'], r['n6']] for r in rows]


def get_power655_numbers():
    """Get Power 6/55 numbers only as list of lists [[n1..n6, bonus], ...]."""
    rows = get_power655_all()
    return [[r['n1'], r['n2'], r['n3'], r['n4'], r['n5'], r['n6'], r['bonus']] for r in rows]


def get_latest_date(lottery_type):
    """Get the latest draw date for a lottery type."""
    conn = get_db()
    table = 'mega645' if lottery_type == 'mega' else 'power655'
    row = conn.execute(f'SELECT MAX(draw_date) as max_date FROM {table}').fetchone()
    conn.close()
    return row['max_date'] if row else None


def get_count(lottery_type):
    """Get total number of draws for a lottery type."""
    conn = get_db()
    table = 'mega645' if lottery_type == 'mega' else 'power655'
    row = conn.execute(f'SELECT COUNT(*) as cnt FROM {table}').fetchone()
    conn.close()
    return row['cnt']


def get_first_date(lottery_type):
    """Get the earliest draw date for a lottery type."""
    conn = get_db()
    table = 'mega645' if lottery_type == 'mega' else 'power655'
    row = conn.execute(f'SELECT MIN(draw_date) as min_date FROM {table}').fetchone()
    conn.close()
    return row['min_date'] if row else None


def export_csv(lottery_type):
    """Export data to CSV file."""
    if lottery_type == 'mega':
        rows = get_mega645_all()
        filename = os.path.join(CSV_DIR, 'mega645.csv')
        headers = ['draw_date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'jackpot']
    else:
        rows = get_power655_all()
        filename = os.path.join(CSV_DIR, 'power655.csv')
        headers = ['draw_date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'bonus', 'jackpot']
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"[CSV] Exported {len(rows)} rows to {filename}")
    return filename


def get_recent(lottery_type, n=20):
    """Get N most recent draws."""
    conn = get_db()
    table = 'mega645' if lottery_type == 'mega' else 'power655'
    rows = conn.execute(
        f'SELECT * FROM {table} ORDER BY draw_date DESC LIMIT ?', (n,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def init_predictions_table():
    """Initialize predictions tracking table."""
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lottery_type TEXT NOT NULL,
        method TEXT NOT NULL,
        n1 INTEGER NOT NULL,
        n2 INTEGER NOT NULL,
        n3 INTEGER NOT NULL,
        n4 INTEGER NOT NULL,
        n5 INTEGER NOT NULL,
        n6 INTEGER NOT NULL,
        target_date TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        matches INTEGER DEFAULT -1,
        actual_n1 INTEGER,
        actual_n2 INTEGER,
        actual_n3 INTEGER,
        actual_n4 INTEGER,
        actual_n5 INTEGER,
        actual_n6 INTEGER
    )''')
    conn.commit()
    conn.close()


def save_prediction(lottery_type, method, numbers, target_date=None):
    """Save a prediction to tracking table."""
    if len(numbers) < 6:
        return
    nums = sorted(numbers[:6])
    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT INTO predictions
                (lottery_type, method, n1, n2, n3, n4, n5, n6, target_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
             (lottery_type, method, nums[0], nums[1], nums[2],
              nums[3], nums[4], nums[5], target_date))
    conn.commit()
    conn.close()


def update_prediction_results(lottery_type):
    """Match predictions against actual results. Call after new data arrives."""
    conn = get_db()
    c = conn.cursor()
    table = 'mega645' if lottery_type == 'mega' else 'power655'
    
    # Get unmatched predictions
    preds = c.execute(
        'SELECT id, n1, n2, n3, n4, n5, n6, target_date FROM predictions '
        'WHERE lottery_type = ? AND matches = -1 AND target_date IS NOT NULL',
        (lottery_type,)
    ).fetchall()
    
    for p in preds:
        pid = p[0]
        pred_set = set(p[1:7])
        target = p[7]
        
        # Find actual result for this target date
        actual = c.execute(
            f'SELECT n1, n2, n3, n4, n5, n6 FROM {table} WHERE draw_date = ?',
            (target,)
        ).fetchone()
        
        if actual:
            actual_set = set(actual)
            match_count = len(pred_set & actual_set)
            c.execute(
                'UPDATE predictions SET matches = ?, '
                'actual_n1 = ?, actual_n2 = ?, actual_n3 = ?, '
                'actual_n4 = ?, actual_n5 = ?, actual_n6 = ? '
                'WHERE id = ?',
                (match_count, actual[0], actual[1], actual[2],
                 actual[3], actual[4], actual[5], pid)
            )
    
    conn.commit()
    conn.close()


def get_predictions_history(lottery_type, limit=50):
    """Get prediction history with match results."""
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM predictions WHERE lottery_type = ? '
        'ORDER BY created_at DESC LIMIT ?',
        (lottery_type, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Auto-init on import
init_db()
init_predictions_table()
