"""Chunked ingestion with per-source atomic replacement and restart safety."""
import json
import time
import pandas as pd
from src.config import DATA_DIR, RAW_CSV_FILES, CHUNK_SIZE, RAW_DTYPES, TABLE_EVENTS
from src.quality import evaluate_data_quality, persist_quality_report
from src.processing import clean_and_transform_chunk
from src.storage import get_connection, init_database, log_pipeline_run
from src.logger import get_logger

logger = get_logger("Ingestion")


def ingest_file(file_name: str, max_chunks=None, run_quality_gate=True):
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    file_path = DATA_DIR / file_name
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    init_database()
    stat = file_path.stat()
    fingerprint = json.dumps([stat.st_size, stat.st_mtime_ns, "etl-v2"])
    conn = get_connection()
    existing = conn.execute("SELECT fingerprint, chunk_limit, rows_loaded FROM ingestion_manifest WHERE source_file=?", (file_name,)).fetchone()
    if existing and existing[0] == fingerprint:
        previous_limit = existing[1]
        if previous_limit is None or previous_limit == max_chunks:
            conn.close()
            logger.info(f"Skipping unchanged source {file_name} ({existing[2]:,} rows already loaded)")
            return {"file": file_name, "status": "SKIPPED", "rows_inserted": 0}
        if max_chunks is not None and previous_limit is not None and max_chunks < previous_limit:
            conn.close()
            raise ValueError("Refusing to replace a larger sample with a smaller sample; use a separate runtime directory")
    start = time.monotonic()
    total_read = total_inserted = duplicates = rejected = 0
    reports = []
    warning = False
    try:
        # TEMP storage is disk-backed; existing source remains visible until final commit.
        conn.execute("PRAGMA temp_store = FILE")
        conn.execute("CREATE TEMP TABLE staged_events AS SELECT * FROM events WHERE 0")
        conn.execute("""CREATE UNIQUE INDEX temp.idx_staged_identity ON staged_events (
            event_time, event_type, product_id, user_id, user_session, price,
            category_id, category_code, brand
        )""")
        with pd.read_csv(file_path, chunksize=CHUNK_SIZE, dtype=RAW_DTYPES) as reader:
            for number, raw in enumerate(reader):
                if max_chunks is not None and number >= max_chunks:
                    break
                total_read += len(raw)
                if run_quality_gate:
                    passed, report = evaluate_data_quality(raw, f"{file_name}_chunk_{number}")
                    reports.append(report)
                    warning |= not passed
                clean, stats = clean_and_transform_chunk(raw, number, file_name)
                # Validate every cleaned chunk; raw warnings are retained for audit.
                if not clean.empty:
                    passed, report = evaluate_data_quality(clean, f"{file_name}_clean_{number}")
                    if not passed:
                        raise ValueError(f"Cleaned data failed quality validation: {report}")
                    columns = list(clean.columns)
                    marks = ",".join("?" for _ in columns)
                    names = ",".join('"' + c + '"' for c in columns)
                    rows = clean.astype(object).where(clean.notna(), None).itertuples(index=False, name=None)
                    before = conn.total_changes
                    conn.executemany(f"INSERT OR IGNORE INTO staged_events ({names}) VALUES ({marks})", rows)
                    inserted = conn.total_changes - before
                    total_inserted += inserted
                    duplicates += len(clean) - inserted
                duplicates += stats["duplicates_removed"]
                rejected += len(raw) - len(clean) - stats["duplicates_removed"]
                if (number + 1) % 5 == 0:
                    logger.info(f"{file_name}: read {total_read:,}, staged {total_inserted:,}")
        # Reject file changes during ingestion, avoiding a misleading completion marker.
        after = file_path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed during ingestion: {file_name}")
        if total_inserted == 0:
            raise ValueError(f"No valid events in {file_name}; previous data preserved")
        conn.execute("DELETE FROM events WHERE source_file=?", (file_name,))
        columns = [r[1] for r in conn.execute("PRAGMA table_info(events)") if r[1] != "event_id"]
        names = ",".join(columns)
        conn.execute(f"INSERT INTO events ({names}) SELECT {names} FROM staged_events")
        conn.execute("""INSERT INTO ingestion_manifest(source_file,fingerprint,chunk_limit,rows_loaded)
            VALUES (?,?,?,?) ON CONFLICT(source_file) DO UPDATE SET
            fingerprint=excluded.fingerprint, chunk_limit=excluded.chunk_limit,
            rows_loaded=excluded.rows_loaded, updated_at=CURRENT_TIMESTAMP""",
            (file_name, fingerprint, max_chunks, total_inserted))
        conn.commit()
    except Exception as exc:
        conn.rollback()
        conn.close()
        log_pipeline_run("ETL_Ingestion", TABLE_EVENTS, file_name, total_read, 0,
                         duplicates, rejected, time.monotonic()-start, "FAILED", str(exc))
        raise
    else:
        conn.close()
    for report in reports:
        persist_quality_report(report, file_name)
    status = "QUALITY_WARNING" if warning else "SUCCESS"
    duration = time.monotonic() - start
    log_pipeline_run("ETL_Ingestion", TABLE_EVENTS, file_name, total_read, total_inserted,
                     duplicates, rejected, duration, status)
    logger.info(f"{file_name}: {total_inserted:,} rows committed, {status}, {duration:.1f}s")
    return {"file": file_name, "rows_read": total_read, "rows_inserted": total_inserted,
            "duplicates_removed": duplicates, "invalid_rows_removed": rejected,
            "duration_seconds": round(duration, 2), "status": status}


def ingest_all_files(max_chunks_per_file=None):
    # Preflight all files before modifying any source.
    missing = [name for name in RAW_CSV_FILES if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files in {DATA_DIR}: {missing}")
    return [ingest_file(name, max_chunks_per_file) for name in RAW_CSV_FILES]
