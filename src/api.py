"""Local JSON HTTP API for the project demo (standard library only)."""
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from src.inference import get_customer_360, batch_inference
from src.monitoring import get_system_health

logger = logging.getLogger("API")


class APIHandler(BaseHTTPRequestHandler):
    def respond(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path.rstrip("/")
        try:
            if path == "/health":
                health = get_system_health()
                ready = health["database"]["total_events"] > 0 and all(health["models"].values()) and health["model_schema_current"]
                self.respond(200 if ready else 503, health)
            elif path.startswith("/customers/"):
                raw_id = path.removeprefix("/customers/")
                if not raw_id.isdigit() or len(raw_id) > 18 or int(raw_id) == 0:
                    self.respond(400, {"error": "user_id must be a positive integer of at most 18 digits"})
                    return
                profile = get_customer_360(int(raw_id))
                self.respond(404 if profile.get("status") == "NOT_FOUND" else 200, profile)
            elif path == "":
                self.respond(200, {"service": "E-Commerce Analytics", "endpoints": [
                    "GET /health", "GET /customers/{user_id}", "POST /batch (JSON: user_ids, maximum 100)"]})
            else:
                self.respond(404, {"error": "Endpoint not found"})
        except Exception:
            logger.exception("API request failed")
            self.respond(503, {"error": "Data or models are unavailable. Run the pipeline and inspect logs."})

    def do_POST(self):
        if urlsplit(self.path).path.rstrip("/") != "/batch":
            self.respond(404, {"error": "Endpoint not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("Request body must contain 1-16384 bytes")
            payload = json.loads(self.rfile.read(length))
            users = payload.get("user_ids") if isinstance(payload, dict) else None
            if not isinstance(users, list) or not 1 <= len(users) <= 100:
                raise ValueError("user_ids must be a list of 1-100 integers")
            if any(type(uid) is not int or not 0 < uid < 10**18 for uid in users):
                raise ValueError("user_ids must contain positive integers of at most 18 digits")
        except (ValueError, UnicodeDecodeError) as exc:
            self.respond(400, {"error": str(exc)})
            return
        try:
            records = batch_inference(users).to_dict(orient="records")
            found = {r["user_id"] for r in records}
            self.respond(200, {"results": records, "not_found": [u for u in users if u not in found]})
        except Exception:
            logger.exception("Batch inference failed")
            self.respond(503, {"error": "Data or models are unavailable. Run the pipeline and inspect logs."})

    def setup(self):
        super().setup()
        self.connection.settimeout(15)


def serve(port=8000):
    server = ThreadingHTTPServer(("127.0.0.1", port), APIHandler)
    print(f"Local API: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
