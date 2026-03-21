"""
NeuroScan — SQLite Scan History Manager
Stores all scan results persistently in neuroscan.db
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'neuroscan.db')


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS scans (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at    TEXT    NOT NULL,
                label         TEXT    NOT NULL,
                cls_name      TEXT    NOT NULL,
                cls_style     TEXT    NOT NULL,
                confidence    REAL    NOT NULL,
                severity      TEXT    NOT NULL,
                reco          TEXT    NOT NULL,
                probabilities TEXT    NOT NULL,
                thumb_b64     TEXT    NOT NULL,
                gradcam_b64   TEXT
            )
        ''')
        conn.commit()
    print(f"Database ready: {DB_PATH}")


def save_scan(label, cls_name, cls_style, confidence, severity,
              reco, probabilities, thumb_b64, gradcam_b64=None):
    """
    Insert a new scan record. Returns the new row id.
    probabilities: dict {class_name: float}
    """
    ts = datetime.now().strftime('%d %b %Y, %H:%M')
    with get_conn() as conn:
        cur = conn.execute('''
            INSERT INTO scans
                (created_at, label, cls_name, cls_style, confidence, severity,
                 reco, probabilities, thumb_b64, gradcam_b64)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ts, label, cls_name, cls_style,
            confidence, severity, reco,
            json.dumps(probabilities),
            thumb_b64, gradcam_b64
        ))
        conn.commit()
        return cur.lastrowid


def get_scans(limit=50):
    """Return the most recent `limit` scans, newest first."""
    with get_conn() as conn:
        rows = conn.execute(
            'SELECT * FROM scans ORDER BY id DESC LIMIT ?', (limit,)
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d['probabilities'] = json.loads(d['probabilities'])
        result.append(d)
    return result


def delete_scan(scan_id: int):
    """Delete a single scan by id."""
    with get_conn() as conn:
        conn.execute('DELETE FROM scans WHERE id = ?', (scan_id,))
        conn.commit()


def clear_all_scans():
    """Delete all scan history."""
    with get_conn() as conn:
        conn.execute('DELETE FROM scans')
        conn.commit()


def get_stats():
    """Return quick stats for the history panel header."""
    with get_conn() as conn:
        total = conn.execute('SELECT COUNT(*) FROM scans').fetchone()[0]
        by_class = conn.execute(
            'SELECT cls_name, COUNT(*) as cnt FROM scans GROUP BY cls_name'
        ).fetchall()
    return {
        'total': total,
        'by_class': {row['cls_name']: row['cnt'] for row in by_class}
    }