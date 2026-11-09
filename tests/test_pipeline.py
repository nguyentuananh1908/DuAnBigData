"""Regression tests use an isolated runtime and never modify project data."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_test_root = Path(__file__).resolve().parents[1] / ".test-runs"
_test_root.mkdir(exist_ok=True)
_runtime = tempfile.mkdtemp(prefix="run-", dir=_test_root)
os.environ["BIGDATA_RUNTIME_DIR"] = _runtime
os.environ.pop("BIGDATA_DATA_DIR", None)
os.environ.pop("BIGDATA_DB_PATH", None)

import pandas as pd
from src.config import DB_PATH, DATA_DIR, OUTPUT_DIR
from src.storage import init_database, get_connection, export_to_parquet
from src.ingestion import ingest_file
from src.features import extract_user_features, extract_single_user_features
from src.processing import clean_and_transform_chunk
from src.ml_models import train_churn_prediction


def event(user=1, time="2020-01-01 00:00:00 UTC", kind="purchase", price=10):
    return dict(event_time=time, event_type=kind, product_id=10,
                category_id="1487580000000000001", category_code="electronics.phone",
                brand="test", price=price, user_id=user, user_session="session-00000001")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        init_database()
        c = get_connection()
        for table in ["events", "ingestion_manifest", "pipeline_metadata", "data_quality_reports"]:
            c.execute(f"DELETE FROM {table}")
        c.commit()
        c.close()

    def load(self, rows, chunk_size=2):
        pd.DataFrame(rows).to_csv(DATA_DIR / "sample.csv", index=False)
        with patch("src.ingestion.CHUNK_SIZE", chunk_size):
            return ingest_file("sample.csv")

    def count(self):
        c = get_connection()
        n = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        c.close()
        return n

    def test_repeat_and_cross_chunk_duplicates(self):
        a, b = event(), event(user=2)
        self.load([a, b, a])
        self.assertEqual(self.count(), 2)
        self.assertEqual(ingest_file("sample.csv")["status"], "SKIPPED")
        self.assertEqual(self.count(), 2)
        self.load([a, event(user=3)])
        self.assertEqual(self.count(), 2)
        self.assertIsNone(extract_single_user_features(2))

    def test_failed_replacement_preserves_previous_data(self):
        self.load([event()])
        bad = event(user="invalid-id")
        with self.assertRaises(ValueError):
            self.load([event(user=2), bad], chunk_size=1)
        self.assertEqual(self.count(), 1)
        self.assertIsNotNone(extract_single_user_features(1))

    def test_sample_can_expand_to_full(self):
        pd.DataFrame([event(user=i) for i in range(4)]).to_csv(DATA_DIR / "sample.csv", index=False)
        with patch("src.ingestion.CHUNK_SIZE", 2):
            ingest_file("sample.csv", max_chunks=1)
            self.assertEqual(self.count(), 2)
            ingest_file("sample.csv")
        self.assertEqual(self.count(), 4)

    def test_revenue_and_future_labels(self):
        self.load([event(1, kind="view", price=999), event(1, price=20),
                   event(2, price=30), event(1, "2020-02-15 00:00:00 UTC", price=100)])
        serving = extract_single_user_features(1)
        self.assertEqual(serving["total_spent"], 120)
        self.assertEqual(serving["avg_spent"], 60)
        descriptive = extract_user_features().set_index("user_id")
        self.assertEqual(descriptive.loc[1, "total_spent"], serving["total_spent"])
        training = extract_user_features(for_churn=True).set_index("user_id")
        self.assertEqual(training.loc[1, "total_spent"], 20)
        self.assertEqual(training.loc[1, "churned"], 0)
        self.assertEqual(training.loc[2, "churned"], 1)

    def test_invalid_rows_and_high_value_purchase(self):
        rows = [event(price=100000), event(price=-1), event(kind="invalid"),
                event(time="not-a-date"), event(user=None), event(time="2020-02-29 23:59:59 UTC")]
        rows[1]["user_session"] = None
        clean, _ = clean_and_transform_chunk(pd.DataFrame(rows))
        self.assertEqual(len(clean), 2)
        self.assertIn(100000, clean.price.tolist())
        self.assertEqual(clean.iloc[0].category_id, "1487580000000000001")

    def test_parquet_matches_database(self):
        self.load([event(), event(user=2)])
        path = export_to_parquet("SELECT * FROM events", OUTPUT_DIR / "test.parquet")
        self.assertEqual(len(pd.read_parquet(path)), self.count())

    def test_churn_rejects_leaky_features(self):
        self.load([event(), event(user=2)])
        with self.assertRaisesRegex(ValueError, "for_churn"):
            train_churn_prediction(extract_user_features())

    def test_every_chunk_has_quality_report(self):
        self.load([event(user=i) for i in range(5)])
        c = get_connection()
        self.assertEqual(c.execute("SELECT COUNT(*) FROM data_quality_reports").fetchone()[0], 3)
        c.close()

    def test_training_and_inference(self):
        from src.ml_models import train_customer_segmentation, compute_recommendations
        from src.inference import get_customer_360
        rows = []
        for uid in range(1, 121):
            for j in range(1 + uid % 7):
                rows.append(event(uid, f"2020-01-{1+j:02d} 00:00:00 UTC",
                                  "purchase" if j % 2 else "view", uid * 1.25))
            if uid % 3 == 0:
                rows.append(event(uid, "2020-02-20 00:00:00 UTC", price=uid * 2))
        self.load(rows, chunk_size=1000)
        train_customer_segmentation(extract_user_features())
        _, metrics, _ = train_churn_prediction(extract_user_features(for_churn=True))
        self.assertIn("baseline_f1", metrics)
        compute_recommendations()
        profile = get_customer_360(3)
        self.assertEqual(profile["status"], "SUCCESS")
        self.assertTrue(0 <= profile["churn_prediction"]["churn_probability"] <= 1)
        self.assertTrue(profile["recommended_products"])
        self.assertEqual(get_customer_360(999999)["status"], "NOT_FOUND")

    def test_invalid_chunk_limit(self):
        with self.assertRaises(ValueError):
            ingest_file("sample.csv", max_chunks=0)

    def test_keeps_full_load_when_sample_requested(self):
        self.load([event(), event(user=2), event(user=3)])
        result = ingest_file("sample.csv", max_chunks=1)
        self.assertEqual(result["status"], "SKIPPED")
        self.assertEqual(self.count(), 3)

    def test_refuses_to_shrink_sample(self):
        pd.DataFrame([event(user=i) for i in range(4)]).to_csv(DATA_DIR / "sample.csv", index=False)
        with patch("src.ingestion.CHUNK_SIZE", 2):
            ingest_file("sample.csv", max_chunks=2)
            with self.assertRaises(ValueError):
                ingest_file("sample.csv", max_chunks=1)
        conn = get_connection()
        loaded = conn.execute("SELECT rows_loaded FROM ingestion_manifest WHERE source_file='sample.csv'").fetchone()
        conn.close()
        self.assertIsNotNone(loaded)


if __name__ == "__main__":
    unittest.main()
