import sqlite3
from datetime import datetime
import os


SHOP_DB_PATH = os.getenv("SHOP_DB_PATH", "shop.db")

def _odb():
    conn = sqlite3.connect(SHOP_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _o_init():
    conn = _odb(); cur = conn.cursor()
    cur.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_no TEXT UNIQUE,                 -- EVA-YYYYMMDD-000123
        customer_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        total REAL NOT NULL DEFAULT 0.0,
        items_json TEXT NOT NULL,
        -- new structured fields:
        first_name TEXT,
        last_name  TEXT,
        city       TEXT,
        np_branch  TEXT,
        phone      TEXT,
        note       TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS order_status_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER NOT NULL,
        old_status TEXT,
        new_status TEXT NOT NULL,
        changed_at TEXT NOT NULL,
        note TEXT,
        FOREIGN KEY(order_id) REFERENCES orders(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
    CREATE INDEX IF NOT EXISTS idx_orders_status   ON orders(status);
    CREATE INDEX IF NOT EXISTS idx_orders_order_no ON orders(order_no);
    """)
    # migrate columns if older schema present
    cur.execute("PRAGMA table_info(orders)")
    cols = {r[1] for r in cur.fetchall()}
    for col, ddl in [
        ("order_no", "ALTER TABLE orders ADD COLUMN order_no TEXT"),
        ("first_name", "ALTER TABLE orders ADD COLUMN first_name TEXT"),
        ("last_name",  "ALTER TABLE orders ADD COLUMN last_name TEXT"),
        ("city",       "ALTER TABLE orders ADD COLUMN city TEXT"),
        ("np_branch",  "ALTER TABLE orders ADD COLUMN np_branch TEXT"),
        ("phone",      "ALTER TABLE orders ADD COLUMN phone TEXT"),
        ("note",       "ALTER TABLE orders ADD COLUMN note TEXT"),
    ]:
        if col not in cols:
            cur.execute(ddl)
    conn.commit(); conn.close()


if __name__=="__main__":
    _o_init()