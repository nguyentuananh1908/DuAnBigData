import sqlite3
from pathlib import Path

conn = sqlite3.connect('data/bigdata.db')
tables = conn.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()

print('Database Tables:')
for t in tables:
    count = conn.execute(f'SELECT COUNT(*) FROM {t[0]}').fetchone()[0]
    print(f'  {t[0]}: {count:,} rows')

db_size = Path('data/bigdata.db').stat().st_size / (1024*1024)
print(f'\n📊 Database Size: {db_size:.2f} MB')

conn.close()
