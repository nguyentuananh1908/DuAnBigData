import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import patch
from src.api import APIHandler


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), APIHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, payload=None):
        request = Request(self.base + path, data=json.dumps(payload).encode() if payload is not None else None)
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as exc:
            response = exc
        with response:
            return response.status, json.load(response)

    def test_routes_and_validation(self):
        self.assertEqual(self.request("/")[0], 200)
        self.assertEqual(self.request("/unknown")[0], 404)
        self.assertEqual(self.request("/customers/abc")[0], 400)
        self.assertEqual(self.request("/batch", {"user_ids": [True]})[0], 400)
        self.assertEqual(self.request("/batch", {"user_ids": list(range(101))})[0], 400)

    def test_customer_status(self):
        with patch("src.api.get_customer_360", return_value={"status": "SUCCESS"}):
            self.assertEqual(self.request("/customers/123")[0], 200)
        with patch("src.api.get_customer_360", return_value={"status": "NOT_FOUND"}):
            self.assertEqual(self.request("/customers/123")[0], 404)

    def test_unavailable_data(self):
        with patch("src.api.get_customer_360", side_effect=RuntimeError("Missing model")):
            self.assertEqual(self.request("/customers/123")[0], 503)
