"""Additional read-only diagnostics: max_seq_id table and migrations."""
import sqlite3

conn = sqlite3.connect("chroma_store/chroma.sqlite3")
conn.row_factory = sqlite3.Row

print("=== MAX_SEQ_ID table schema ===")
schema = conn.execute("PRAGMA table_info(max_seq_id)").fetchall()
for row in schema:
    print(dict(row))

print()
print("=== MAX_SEQ_ID contents ===")
rows = conn.execute("SELECT *, typeof(seq_id) FROM max_seq_id").fetchall()
for r in rows:
    val = r["seq_id"]
    print(f"  segment_id={r['segment_id']}  seq_id={repr(val)}  sqlite_type={r['typeof(seq_id)']}  python_type={type(val).__name__}")

print()
print("=== MIGRATIONS table ===")
try:
    migrations = conn.execute("SELECT * FROM migrations ORDER BY applied_at").fetchall()
    for m in migrations:
        print(dict(m))
except Exception as e:
    print("ERROR:", e)

print()
print("=== EMBEDDINGS_QUEUE_CONFIG ===")
try:
    rows = conn.execute("SELECT * FROM embeddings_queue_config").fetchall()
    for r in rows:
        print(dict(r))
except Exception as e:
    print("ERROR:", e)

conn.close()
print("\n=== DONE (no changes made) ===")
