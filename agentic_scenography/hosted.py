
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
import webbrowser
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import jwt
from openai import APIError, OpenAI

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
_TERMINAL_REFRESH_ERRORS = {
    "invalid_grant", "invalid_refresh_token", "token_expired",
    "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused",
}


class ConnectionError(RuntimeError):
    """A sanitized connection error, safe to show in notebook output."""


class SignInRequired(ConnectionError):
    """A user must complete sign-in before another request can start."""


class ModelRequestError(ConnectionError):
    """No completed model response was received; no tool should be executed."""


class _TokenError(ConnectionError):
    def __init__(self, code, status=None):
        self.code = _safe_code(code)
        self.status = status
        super().__init__(f"Token request failed ({self.code}, HTTP {status}).")


def _safe_code(value):
    # Error descriptions can contain request data. Expose only short error codes.
    return value if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", value) else "unknown_error"


def _read_json(path):
    if not path.exists():
        return {}
    if path.is_symlink():
        raise ConnectionError("The account file must not be a symbolic link.")
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (OSError, ValueError):
        raise ConnectionError("The local account file could not be read.") from None


def _atomic_json(path, value):
    """Atomic replacement prevents a partial credential file after interruption."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary_path = stream.name
            os.chmod(temporary_path, 0o600)
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


@contextmanager
def _file_lock(path):
    """Serialize rotating refresh tokens across notebook kernels/processes."""
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.write(b"0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _callback_values(path, expected_state, saved_client_id):
    parsed = urlparse(path)
    values = parse_qs(parsed.query, keep_blank_values=True)
    if parsed.path != "/auth/callback" or any(len(v) != 1 for v in values.values()):
        raise ConnectionError("Malformed sign-in callback.")
    returned_state = values.get("state", [""])[0]
    if not secrets.compare_digest(returned_state, expected_state):
        raise ConnectionError("Sign-in state did not match.")
    result = {key: value[0] for key, value in values.items()}
    if result.get("error"):
        return {"error": _safe_code(result["error"])}
    client_id = result.get("client_id") or saved_client_id
    if not client_id or client_id == "dynamic_agent_client":
        raise ConnectionError("No registered client ID was returned.")
    if saved_client_id and client_id != saved_client_id:
        raise ConnectionError("The returned client ID did not match.")
    if not result.get("code"):
        raise ConnectionError("No authorization code was returned.")
    return {"client_id": client_id, "code": result["code"]}


def _token_request(parameters):
    request = Request(
        ISSUER + "/api/accounts/oauth/token",
        data=urlencode(parameters).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        try:
            body = json.loads(error.read())
            code = body.get("error", "unknown_error")
            if isinstance(code, dict):
                code = code.get("code")
        except (ValueError, OSError):
            code = "unknown_error"
        raise _TokenError(code, error.code) from None
    except (URLError, TimeoutError, OSError, ValueError):
        raise ConnectionError("The token service could not be reached. Credentials were preserved; try again later.") from None


def _validate_identity(id_token, client_id, expected_subject=None, nonce=None):
    try:
        key = jwt.PyJWKClient(ISSUER + "/.well-known/jwks.json").get_signing_key_from_jwt(id_token)
        required = ["iss", "aud", "sub", "exp", "iat"] + (["nonce"] if nonce else [])
        identity = jwt.decode(
            id_token, key.key, algorithms=["RS256", "ES256"], audience=client_id,
            issuer=ISSUER, leeway=5, options={"require": required},
        )
        if nonce and not secrets.compare_digest(str(identity["nonce"]), nonce):
            raise ValueError
        if expected_subject and identity["sub"] != expected_subject:
            raise ValueError
        return identity
    except Exception:
        raise ConnectionError("Sign-in identity verification failed. No model request was made.") from None


def _namespace_tools(tools):
    if not tools:
        return None
    flat = [tool for tool in tools if tool.get("type") in ("function", "custom")]
    groups = [tool for tool in tools if tool.get("type") == "namespace"]
    if len(flat) + len(groups) != len(tools):
        raise ValueError("This project supports local function/custom tools only.")
    if flat:
        groups = groups + [{"type": "namespace", "name": "scenography", "description": "Local scenography and Blender tools.", "tools": flat}]
    return groups


def function_calls(response):
    """Parse requests only; actual tool execution belongs to the notebook graph."""
    calls = []
    for item in response.get("output", []):
        if item.get("type") != "function_call":
            continue
        try:
            arguments = json.loads(item["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError
            calls.append({"name": item["name"], "namespace": item.get("namespace"), "arguments": arguments, "call_id": item["call_id"]})
        except (KeyError, TypeError, ValueError):
            raise ModelRequestError("The model returned malformed function arguments; no tool was executed.") from None
    return calls


class ChatGPTConnection:
    """Use the reader's own ChatGPT account, independent of this author's chat.

    Call sign_in() explicitly. Normal requests never open a browser. Named profiles
    keep separate accounts/workspaces apart; 'default' reuses the pilot registration.
    Set persist_tokens=False for an in-memory session. No sign-in occurs on import.
    """

    def __init__(self, config_dir=None, profile="default", persist_tokens=True, timeout=300):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", profile):
            raise ValueError("Use a short profile name containing letters, digits, '-' or '_'.")
        self.config_dir = Path(config_dir or Path.home() / ".config" / "agentic-scenography")
        self.profile = profile
        prefix = "chatgpt" if profile == "default" else f"chatgpt-{profile}"
        self.registration_file = self.config_dir / f"{prefix}-registration.json"
        self.credentials_file = self.config_dir / f"{prefix}-credentials.json"
        self.persist_tokens = persist_tokens
        self.timeout = timeout
        self._registration = {}
        self._credentials = {}
        self._thread_lock = threading.RLock()

    def __repr__(self):
        return f"ChatGPTConnection(profile={self.profile!r}, signed_in={bool(self._credentials.get('access_token'))})"

    @property
    def account(self):
        return {"profile": self.profile, "email": self._registration.get("email"), "signed_in": bool(self._credentials.get("access_token"))}

    @contextmanager
    def _locked(self):
        self.config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.config_dir.is_symlink():
            raise ConnectionError("The account directory must not be a symbolic link.")
        with self._thread_lock, _file_lock(self.config_dir / f".{self.profile}.lock"):
            yield

    def _load(self):
        self._registration = _read_json(self.registration_file)
        if self.persist_tokens:
            self._credentials = _read_json(self.credentials_file)
        if self._credentials:
            if any(self._credentials.get(key) != self._registration.get(key) for key in ("client_id", "subject")):
                raise ConnectionError("The saved account and credential records do not match.")
            if self._credentials.get("issuer") != ISSUER:
                raise ConnectionError("The saved credential issuer does not match.")

    def _store_tokens(self, tokens, identity, client_id, old=None):
        old = old or {}
        scopes = tokens.get("scope", " ".join(old.get("scopes", []))).split()
        if PLAN_SCOPE not in scopes:
            raise ConnectionError("Sign-in succeeded, but ChatGPT plan usage was not granted.")
        if not tokens.get("access_token") or tokens.get("token_type", "Bearer").lower() != "bearer":
            raise ConnectionError("The token response did not contain a bearer credential.")
        try:
            lifetime = float(tokens["expires_in"])
            if lifetime <= 0:
                raise ValueError
        except (KeyError, ValueError, TypeError):
            raise ConnectionError("The token response did not contain a valid expiry.") from None
        credentials = {
            "issuer": ISSUER, "client_id": client_id, "subject": identity["sub"],
            "access_token": tokens["access_token"], "refresh_token": tokens.get("refresh_token", old.get("refresh_token")),
            "id_token": tokens.get("id_token", old.get("id_token")), "scopes": scopes,
            "expires_at": time.time() + lifetime,
        }
        self._registration.update({"client_id": client_id, "subject": identity["sub"], "email": identity.get("email", self._registration.get("email"))})
        _atomic_json(self.registration_file, self._registration)
        if self.persist_tokens:
            _atomic_json(self.credentials_file, credentials)
        self._credentials = credentials

    def _refresh_if_needed(self):
        credentials = self._credentials
        if credentials.get("access_token") and credentials.get("expires_at", 0) > time.time() + 60:
            if PLAN_SCOPE not in credentials.get("scopes", []):
                raise SignInRequired("ChatGPT plan permission is missing. Run connection.sign_in(force=True).")
            return
        if not credentials.get("refresh_token"):
            raise SignInRequired("Run connection.sign_in() to authorize this notebook.")
        try:
            tokens = _token_request({"grant_type": "refresh_token", "client_id": credentials["client_id"], "refresh_token": credentials["refresh_token"], "resource": RESOURCE})
        except _TokenError as error:
            if error.code in _TERMINAL_REFRESH_ERRORS:
                self._credentials = {}
                if self.persist_tokens:
                    _atomic_json(self.credentials_file, {})
                raise SignInRequired("The saved session expired. Run connection.sign_in() again.") from None
            raise
        identity = {"sub": credentials["subject"]}
        if tokens.get("id_token"):
            identity = _validate_identity(tokens["id_token"], credentials["client_id"], credentials["subject"])
        self._store_tokens(tokens, identity, credentials["client_id"], old=credentials)

    def _new_client(self):
        with self._locked():
            self._load()
            self._refresh_if_needed()
            return OpenAI(api_key=self._credentials["access_token"], base_url=RESOURCE, max_retries=0, timeout=self.timeout)

    def sign_in(self, force=False, timeout=300):
        """Reuse/refresh a protected session, or open the system browser for consent."""
        with self._locked():
            self._load()
            if not force:
                try:
                    self._refresh_if_needed()
                    print("Saved ChatGPT connection is ready. No model request was made.")
                    return self.account
                except SignInRequired:
                    pass
            if not self._registration.get("host_id"):
                default = _read_json(self.config_dir / "chatgpt-registration.json")
                self._registration["host_id"] = default.get("host_id") or f"urn:uuid:{uuid.uuid4()}"
                _atomic_json(self.registration_file, self._registration)
            registration = dict(self._registration)
            previous_tokens = dict(self._credentials)

        client_id = registration.get("client_id")
        state, nonce, verifier = (secrets.token_urlsafe(32), secrets.token_urlsafe(32), secrets.token_urlsafe(64))
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        received = {}
        authorization_url = ""
        start_path = "/auth/start/" + secrets.token_urlsafe(16)

        class CallbackHandler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # Callback URLs and authorization hints must never be logged.

            def do_GET(self):
                if self.path == start_path:
                    self.send_response(302)
                    self.send_header("Location", authorization_url)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Referrer-Policy", "no-referrer")
                    self.end_headers()
                    return
                try:
                    values = _callback_values(self.path, state, client_id)
                except ConnectionError:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Invalid callback. Return to the notebook.")
                    return
                received.update(values)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(b"Authorization received. Return to Jupyter for verification.")

        server = HTTPServer(("127.0.0.1", 0), CallbackHandler)
        server.timeout = 0.5
        redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
        parameters = {"client_id": client_id or "dynamic_agent_client", "ext_agent_host_id": registration["host_id"], "response_type": "code", "redirect_uri": redirect_uri, "scope": SCOPES, "resource": RESOURCE, "state": state, "nonce": nonce, "code_challenge_method": "S256", "code_challenge": challenge}
        if client_id:
            if previous_tokens.get("id_token"):
                parameters["id_token_hint"] = previous_tokens["id_token"]
            if registration.get("email"):
                parameters["login_hint"] = registration["email"]
        else:
            parameters["agent_name_hint"] = "Agentic Scenography"
        authorization_url = ISSUER + "/api/accounts/authorize?" + urlencode(parameters)
        # This local link is safe to print; the authorization URL may hold an ID-token hint.
        local_url = f"http://127.0.0.1:{server.server_port}{start_path}"
        try:
            print(f"Continue with ChatGPT: {local_url}")
            print("Complete sign-in in your browser; then return here.")
            webbrowser.open(local_url)
            deadline = time.monotonic() + timeout
            while not received and time.monotonic() < deadline:
                server.handle_request()
        finally:
            server.server_close()
        if not received:
            raise SignInRequired("Sign-in timed out. Run this cell again for a fresh link.")
        if received.get("error"):
            raise SignInRequired("Authorization was not completed. No model request was made.")
        issued_id = received["client_id"]
        with self._locked():
            # Retain a newly issued ID even if its one-use authorization code expires.
            self._registration = registration | {"client_id": issued_id}
            _atomic_json(self.registration_file, self._registration)
            tokens = _token_request({"grant_type": "authorization_code", "client_id": issued_id, "code": received["code"], "code_verifier": verifier, "redirect_uri": redirect_uri, "resource": RESOURCE})
            identity = _validate_identity(tokens.get("id_token", ""), issued_id, registration.get("subject"), nonce)
            self._store_tokens(tokens, identity, issued_id)
        print("Sign-in verified. ChatGPT plan usage is authorized. No model request was made.")
        return self.account

    def list_models(self):
        """Account-specific model choices, preserving server order."""
        try:
            with self._new_client() as client:
                catalog = client.get("/models", cast_to=dict)
        except APIError as error:
            raise ConnectionError(f"Model listing failed (HTTP {getattr(error, 'status_code', None)}).") from None
        if not isinstance(catalog.get("models"), list):
            raise ConnectionError("The model catalog did not contain the expected models list.")
        return [model for model in catalog["models"] if model.get("visibility") == "list"]

    async def respond(self, model, instructions, input_items, tools=None, reasoning_effort=None):
        """Generate one response without blocking Jupyter's event loop.

        Pass the returned output items back in input_items for tool continuation,
        followed by function_call_output items with the corresponding call_id.
        No tools are executed here, and no conversation is stored by this helper.
        """
        return await asyncio.to_thread(self._respond, model, instructions, input_items, tools, reasoning_effort)

    def _respond(self, model, instructions, input_items, tools=None, reasoning_effort=None):
        if not isinstance(input_items, list):
            raise TypeError("input_items must be a list containing this request's context.")
        arguments = {"model": model, "instructions": instructions, "input": input_items, "store": False, "stream": True, "include": ["reasoning.encrypted_content"]}
        grouped_tools = _namespace_tools(tools)
        if grouped_tools:
            arguments["tools"] = grouped_tools
        if reasoning_effort is not None:
            arguments["reasoning"] = {"effort": reasoning_effort}
        started = time.perf_counter()
        completed = None
        completed_items = {}
        try:
            with self._new_client() as client:
                with client.responses.create(**arguments) as stream:
                    for event in stream:
                        if event.type == "response.output_item.done":
                            item = event.item.model_dump(mode="json", exclude_none=True)
                            completed_items[event.output_index] = item
                        elif event.type == "response.completed":
                            completed = event.response.model_dump(mode="json", exclude_none=True)
                        elif event.type == "response.failed":
                            error = getattr(event.response, "error", None)
                            raise ModelRequestError(f"Response failed: {_safe_code(getattr(error, 'code', None))}. No completed output is available.")
                        elif event.type == "response.incomplete":
                            details = getattr(event.response, "incomplete_details", None)
                            raise ModelRequestError(f"Response incomplete: {_safe_code(getattr(details, 'reason', None))}. No tool was executed.")
                        elif event.type == "error":
                            raise ModelRequestError(f"Response stream error: {_safe_code(getattr(event, 'code', None))}.")
        except APIError as error:
            raise ModelRequestError(f"Model request failed (HTTP {getattr(error, 'status_code', None)}, {_safe_code(getattr(error, 'code', None))}). No automatic retry was made.") from None
        if completed is None:
            raise ModelRequestError("The stream ended without response.completed. No tool was executed.")
        # Some streamed completion events have output=[]: the completed items
        # arrived earlier as output_item.done events. Retain both forms, ordered
        # by output_index, with done items supplying complete reasoning/tool data.
        indexed_output = dict(enumerate(completed.get("output", [])))
        indexed_output.update(completed_items)
        output = [indexed_output[index] for index in sorted(indexed_output)]
        text = "\n".join(part["text"] for item in output if item.get("type") == "message" for part in item.get("content", []) if part.get("type") == "output_text")
        return {"id": completed.get("id"), "model": completed.get("model", model), "status": "completed", "output": output, "text": text, "usage": completed.get("usage", {}), "elapsed": round(time.perf_counter() - started, 3)}
