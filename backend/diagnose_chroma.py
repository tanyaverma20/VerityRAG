"""
Read-only diagnostic of backend/chroma_store/chroma.sqlite3.
No modifications made.
"""
import sqlite3
import os

DB_PATH = "chroma_store/chroma.sqlite3"

print(f"File: {DB_PATH}")
print(f"Size: {os.path.getsize(DB_PATH):,} bytes")
print()

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# 1. All tables
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
print("=== TABLES ===")
for t in tables:
    print(" -", t["name"])
print()

# 2. Collections
print("=== COLLECTIONS ===")
try:
    cols = conn.execute("SELECT * FROM collections").fetchall()
    for c in cols:
        print(dict(c))
except Exception as e:
    print("ERROR:", e)
print()

# 3. Embeddings queue table - where seq_id lives
print("=== EMBEDDINGS_QUEUE schema ===")
try:
    schema = conn.execute("PRAGMA table_info(embeddings_queue)").fetchall()
    for row in schema:
        print(dict(row))
except Exception as e:
    print("ERROR:", e)
print()

# 4. Sample seq_id values and their Python types
print("=== EMBEDDINGS_QUEUE sample seq_ids (type + value) ===")
try:
    rows = conn.execute("SELECT seq_id, typeof(seq_id) FROM embeddings_queue LIMIT 10").fetchall()
    for r in rows:
        val = r[0]
        print(f"  value={repr(val)!s:40s}  sqlite_type={r[1]}  python_type={type(val).__name__}")
    print(f"  ... total count: {conn.execute('SELECT count(*) FROM embeddings_queue').fetchone()[0]}")
except Exception as e:
    print("ERROR:", e)
print()

# 5. max seq_id
print("=== MAX SEQ_ID ===")
try:
    max_row = conn.execute("SELECT max(seq_id), typeof(max(seq_id)) FROM embeddings_queue").fetchone()
    print(f"  max_seq_id value={repr(max_row[0])}  sqlite_type={max_row[1]}  python_type={type(max_row[0]).__name__}")
except Exception as e:
    print("ERROR:", e)
print()

# 6. Embeddings table chunk count
print("=== EMBEDDINGS count ===")
try:
    count = conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    print(f"  Total embeddings: {count}")
    # sample
    sample = conn.execute("SELECT id, seq_id, typeof(seq_id) FROM embeddings LIMIT 5").fetchall()
    for r in sample:
        print(f"  id={r[0]!s:40s}  seq_id={repr(r[1])}  sqlite_type={r[2]}  python_type={type(r[1]).__name__}")
except Exception as e:
    print("ERROR:", e)
print()

# 7. Check segments table
print("=== SEGMENTS ===")
try:
    segs = conn.execute("SELECT * FROM segments").fetchall()
    for s in segs:
        print(dict(s))
except Exception as e:
    print("ERROR:", e)
print()

# 8. Check metadata version/migration info
print("=== MIGRATION_INFO (if exists) ===")
try:
    info = conn.execute("SELECT * FROM migration_info").fetchall()
    for r in info:
        print(dict(r))
except Exception as e:
    print("TABLE NOT FOUND:", e)

print("\n=== DONE (no changes made) ===")
conn.close()
