"""Local HTTP API for the web UI, with `user` and `admin` bearer tokens.

    uv run python -m scriptorium.api            # serves http://127.0.0.1:8765

Security posture (each item has a test):
  * Binds to loopback only; the CLI refuses any other host. Requests whose Host header is
    not localhost/127.0.0.1 are rejected (DNS-rebinding defence).
  * Bearer tokens in the Authorization header, no cookies, so there is no CSRF surface.
    Tokens are created and revoked only with `python -m scriptorium.auth`; this API has no
    endpoint that mints or lists tokens.
  * Roles: `user` can classify, map, submit corrections and manage its own pending ones.
    `admin` can also review, approve, apply, roll back and reset.
  * Every request body is validated against a strict schema (unknown fields rejected,
    lengths capped) and capped at 16 KB. Validation errors do not echo the input.
  * Unexpected errors return a generic 500; details go to the server log only.
  * Strict security headers on every response; no CORS; interactive docs are disabled.
  * Repeated failed authentication triggers a temporary lockout; submissions are
    rate-limited per token.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from starlette.middleware.trustedhost import TrustedHostMiddleware

from scriptorium import auth
from scriptorium import feedback as fb
from scriptorium.auth import AuthError, Principal
from scriptorium.db import Database, DatabaseError
from scriptorium.feedback import ConflictError, FeedbackError, NotFoundError
from scriptorium.ratelimit import RateLimiter
from scriptorium.service import Service, ServiceError

log = logging.getLogger("scriptorium.api")

MAX_BODY_BYTES = 16 * 1024
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
SUBMITS_PER_MINUTE = 20
MAX_FAILED_AUTH = 10
LOCKOUT_SECONDS = 60.0

SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=()",
}


# ---- request schemas -----------------------------------------------------------------------

Catalog = Annotated[StrictStr, Field(pattern=r"^[a-z0-9_-]{1,32}$")]
Text = Annotated[StrictStr, Field(min_length=1, max_length=4000)]
Reason = Annotated[StrictStr, Field(max_length=500)] | None
FeedbackId = Annotated[StrictStr, Field(pattern=r"^fb_[0-9a-f]{12}$")]
TopK = Annotated[StrictInt, Field(ge=1, le=10)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassifyIn(Strict):
    text: Text
    catalog: Catalog = "nist"
    top_k: TopK = 3


class MapRequirementIn(Strict):
    text: Text
    to_catalog: Catalog
    top_k: TopK = 5


class MapControlIn(Strict):
    control_id: Annotated[StrictStr, Field(min_length=1, max_length=24)]
    from_catalog: Catalog
    to_catalog: Catalog
    top_k: TopK = 5


class SubmitIn(Strict):
    text: Text
    label: Annotated[StrictStr, Field(min_length=1, max_length=32)]
    catalog: Catalog = "nist"


class ReasonIn(Strict):
    catalog: Catalog = "nist"
    reason: Reason = None


class ApplyIn(Strict):
    catalog: Catalog = "nist"
    weight: Annotated[float, Field(gt=0, le=100)] = fb.CORRECTION_WEIGHT
    tolerance: Annotated[float, Field(ge=0, le=1)] = fb.GATE_TOLERANCE
    force: bool = False
    reason: Reason = None


class PreviewIn(Strict):
    catalog: Catalog = "nist"
    weight: Annotated[float, Field(gt=0, le=100)] = fb.CORRECTION_WEIGHT
    tolerance: Annotated[float, Field(ge=0, le=1)] = fb.GATE_TOLERANCE


class RollbackIn(Strict):
    catalog: Catalog = "nist"
    ids: Annotated[list[FeedbackId], Field(min_length=1, max_length=50)]
    reason: Reason = None


class ResetIn(Strict):
    catalog: Catalog
    confirm: StrictStr  # must equal the catalog name
    reason: Reason = None


# ---- app -----------------------------------------------------------------------------------


class BodyLimit:
    """Reject request bodies over `limit` bytes, whether or not Content-Length is sent."""

    class _TooLarge(Exception):
        pass

    def __init__(self, app, limit: int = MAX_BODY_BYTES):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and (not length.isdigit() or int(length) > self.limit):
            return await self._reject(send)
        total, started, rejected = 0, False, False

        async def limited_receive():
            nonlocal total, started, rejected
            msg = await receive()
            if msg["type"] == "http.request":
                total += len(msg.get("body", b""))
                if total > self.limit:
                    # Answer now: the framework would turn an exception raised while it
                    # reads the body into its own 400, so anything it sends afterwards is dropped.
                    if not started:
                        await self._reject(send)
                        started = rejected = True
                    raise self._TooLarge
            return msg

        async def tracking_send(msg):
            nonlocal started
            if rejected:
                return
            started = started or msg["type"] == "http.response.start"
            await send(msg)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except self._TooLarge:
            pass

    @staticmethod
    async def _reject(send):
        body = json.dumps({"detail": "request body too large"}).encode()
        await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


class SecurityHeaders:
    """Add the security headers to every response, including errors raised by other middleware."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_headers(msg):
            if msg["type"] == "http.response.start":
                present = {k.lower() for k, _ in msg.get("headers", [])}
                msg = {**msg, "headers": list(msg.get("headers", [])) + [
                    (k.lower().encode(), v.encode()) for k, v in SECURITY_HEADERS.items() if k.lower().encode() not in present
                ]}
            await send(msg)

        await self.app(scope, receive, send_with_headers)


def create_app(service: Service, db: Database, *, clock=time.monotonic) -> FastAPI:
    if service.feedback is None:
        raise ValueError("the API needs a Service with a feedback store")
    store = service.feedback
    app = FastAPI(title="Scriptorium", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    app.add_middleware(BodyLimit)
    app.add_middleware(SecurityHeaders)

    failures: dict[str, list[float]] = {}
    submit_limiters: dict[str, RateLimiter] = {}

    def client_key(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def locked_out(key: str) -> bool:
        recent = [t for t in failures.get(key, []) if clock() - t < LOCKOUT_SECONDS]
        failures[key] = recent
        return len(recent) >= MAX_FAILED_AUTH

    def current(request: Request) -> Principal:
        key = client_key(request)
        if locked_out(key):
            raise HTTPException(429, "too many failed attempts; try again later", headers={"Retry-After": str(int(LOCKOUT_SECONDS))})
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        principal = auth.authenticate(db, token) if scheme.lower() == "bearer" else None
        if principal is None:
            failures.setdefault(key, []).append(clock())
            log.warning("failed authentication from %s for %s %s", key, request.method, request.url.path)
            raise HTTPException(401, "invalid or missing token", headers={"WWW-Authenticate": "Bearer"})
        return principal

    def admin(p: Principal = Depends(current)) -> Principal:
        if not p.is_admin:
            raise HTTPException(403, "admin role required")
        return p

    # ---- error handling ---------------------------------------------------------------------

    def jerr(status: int, detail: Any) -> JSONResponse:
        return JSONResponse({"detail": detail}, status_code=status, headers=SECURITY_HEADERS)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Report where and what, never the submitted value.
        return jerr(422, [{"field": ".".join(str(x) for x in e["loc"][1:]) or str(e["loc"][0]), "problem": e["msg"]} for e in exc.errors()])

    @app.exception_handler(NotFoundError)
    async def not_found(request: Request, exc: NotFoundError):
        return jerr(404, str(exc))

    @app.exception_handler(ConflictError)
    async def conflict(request: Request, exc: ConflictError):
        return jerr(409, str(exc))

    @app.exception_handler(ServiceError)
    @app.exception_handler(FeedbackError)
    @app.exception_handler(AuthError)
    @app.exception_handler(DatabaseError)
    async def bad_request(request: Request, exc: Exception):
        return jerr(400, str(exc))

    @app.exception_handler(Exception)
    async def crash(request: Request, exc: Exception):
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return jerr(500, "internal error")

    # ---- open ------------------------------------------------------------------------------

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    # ---- any authenticated user ------------------------------------------------------------

    @app.get("/api/me")
    def me(p: Principal = Depends(current)):
        return {"name": p.name, "role": p.role}

    @app.get("/api/catalogs")
    def catalogs(p: Principal = Depends(current)):
        return service.list_catalogs()

    @app.post("/api/classify")
    def classify(body: ClassifyIn, p: Principal = Depends(current)):
        return service.classify(body.text, body.catalog, body.top_k)

    @app.post("/api/map/requirement")
    def map_requirement(body: MapRequirementIn, p: Principal = Depends(current)):
        return service.map_requirement(body.text, body.to_catalog, body.top_k)

    @app.post("/api/map/control")
    def map_control(body: MapControlIn, p: Principal = Depends(current)):
        return service.map_control(body.control_id, body.from_catalog, body.to_catalog, body.top_k)

    @app.post("/api/feedback", status_code=201)
    def submit_feedback(body: SubmitIn, p: Principal = Depends(current)):
        limiter = submit_limiters.setdefault(p.token_id, RateLimiter(SUBMITS_PER_MINUTE, clock=clock))
        if not limiter.allow():
            raise HTTPException(429, f"too many submissions; limit is {SUBMITS_PER_MINUTE} per minute", headers={"Retry-After": "60"})
        return fb.submit(service, store, body.catalog, body.text, body.label, p.actor)

    @app.get("/api/feedback")
    def list_feedback(p: Principal = Depends(current), catalog: Catalog = "nist", status: str | None = None, limit: int = 50):
        name = service._catalog(catalog).name
        own = None if p.is_admin else p.actor
        return {"catalog": name, "entries": fb.list_entries(store, name, status, limit, own)}

    @app.post("/api/feedback/{feedback_id}/withdraw")
    def withdraw(feedback_id: str, body: ReasonIn, p: Principal = Depends(current)):
        name = service._catalog(body.catalog).name
        return fb.transition(store, name, feedback_id, "withdrawn", p.actor, body.reason, None if p.is_admin else p.actor)

    # ---- admin only ------------------------------------------------------------------------

    @app.get("/api/admin/feedback/{feedback_id}")
    def admin_get(feedback_id: str, p: Principal = Depends(admin), catalog: Catalog = "nist"):
        name = service._catalog(catalog).name
        if not fb._ID_RE.match(feedback_id):
            raise FeedbackError(fb.BAD_ID)
        entry = store.entries(name).get(feedback_id)
        if entry is None:
            raise NotFoundError(f"no feedback entry {feedback_id} in {name}")
        return entry

    @app.post("/api/admin/feedback/{feedback_id}/approve")
    def approve(feedback_id: str, body: ReasonIn, p: Principal = Depends(admin)):
        return fb.transition(store, service._catalog(body.catalog).name, feedback_id, "approved", p.actor, body.reason)

    @app.post("/api/admin/feedback/{feedback_id}/reject")
    def reject(feedback_id: str, body: ReasonIn, p: Principal = Depends(admin)):
        return fb.transition(store, service._catalog(body.catalog).name, feedback_id, "rejected", p.actor, body.reason)

    @app.post("/api/admin/apply/preview")
    def apply_preview(body: PreviewIn, p: Principal = Depends(admin)):
        return fb.apply(service, store, service._catalog(body.catalog).name, p.actor, body.weight, body.tolerance, dry_run=True)

    @app.post("/api/admin/apply")
    def apply(body: ApplyIn, p: Principal = Depends(admin)):
        return fb.apply(service, store, service._catalog(body.catalog).name, p.actor, body.weight, body.tolerance, body.force, body.reason)

    @app.post("/api/admin/rollback")
    def rollback(body: RollbackIn, p: Principal = Depends(admin)):
        return fb.rollback(store, service._catalog(body.catalog).name, list(body.ids), p.actor, body.reason)

    @app.post("/api/admin/reset")
    def reset(body: ResetIn, p: Principal = Depends(admin)):
        name = service._catalog(body.catalog).name
        if body.confirm != name:
            raise FeedbackError(f"to reset, set confirm to the catalog name ({name})")
        return fb.reset(store, name, p.actor, body.reason or "reset")

    @app.get("/api/admin/metrics")
    def metrics(p: Principal = Depends(admin), catalog: Catalog = "nist"):
        name = service._catalog(catalog).name
        m = service._model(name)
        return {
            "catalog": name,
            "model_version": m.version,
            "heldout": service.heldout_scores(name, m.pipeline),
            "feedback_by_status": dict(Counter(e["status"] for e in store.entries(name).values())),
            "applied_corrections": len(m.corrections),
        }

    @app.get("/api/admin/integrity")
    def integrity(p: Principal = Depends(admin)):
        v = db.verify()
        return {k: v[k] for k in ("ok", "events", "head_seq", "head_hash", "anchors_checked", "problems")}

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if args.host not in LOOPBACK:
        print("error: this server only binds to the loopback interface (127.0.0.1)", file=sys.stderr)
        return 2
    service = Service.default()
    app = create_app(service, service.feedback.db)
    uvicorn.run(app, host=args.host, port=args.port, server_header=False, date_header=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
