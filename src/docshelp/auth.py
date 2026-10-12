"""Who may do what on the DocsHelp server: authentication and authorization for LangGraph Server (lesson 8.10).

`langgraph.auth.json` points the server at `auth` below (`"auth": {"path": ...}`); `uv run poe server-auth`
serves it. Then every API request goes through two steps:

  1. **Authentication** (`@auth.authenticate`): who is calling? We read the `Authorization: Bearer <token>`
     header and verify the token. Whatever we return becomes `ctx.user` in the handlers below and
     `runtime.user` in graph factories.
  2. **Authorization** (`@auth.on...`): may this user do this action on this resource? A handler can refuse
     (raise 403), stamp the new resource (e.g. `metadata["owner"]`), rewrite the request, and return a
     *filter*: the server then only shows the caller resources whose metadata match it.

The auth provider: real apps get tokens from an identity provider (Supabase, Auth0, Clerk, Entra ID,
Cognito...) and verify its signature with the provider's public keys (JWKS). To stay free and offline,
`issue_token` plays the provider here: it signs a standard JWT (HS256) with a secret from the environment
(`DOCSHELP_AUTH_SECRET`). Swapping in a real provider changes only `verify_token`.
"""

from __future__ import annotations

import time
from typing import Literal

import jwt
from langgraph_sdk import Auth

from docshelp.config import Settings
from docshelp.customers import ACCOUNTS

Role = Literal["customer", "staff"]
ALGORITHM = "HS256"


# --- The auth provider stand-in ---------------------------------------------------------------------------


def issue_token(subject: str, role: Role = "customer", *, ttl_seconds: int = 3600, secret: str | None = None) -> str:
    """Sign in as `subject` (a customer's account id, or a staff member's name): what a login page returns."""
    if role == "customer" and subject not in ACCOUNTS:
        raise ValueError(f"unknown account {subject!r}")
    now = int(time.time())
    claims = {"sub": subject, "role": role, "iat": now, "exp": now + ttl_seconds}
    return jwt.encode(claims, secret or Settings.from_env().auth_secret, algorithm=ALGORITHM)


def verify_token(token: str, *, secret: str | None = None) -> dict:
    """The claims of a valid token. Raises `jwt.InvalidTokenError` (bad signature, expired, malformed)."""
    return jwt.decode(token, secret or Settings.from_env().auth_secret, algorithms=[ALGORITHM])


def user_for(claims: dict) -> dict:
    """The user the server will see as `ctx.user`: `identity` is required, the rest is up to us."""
    role = claims.get("role", "customer")
    return {"identity": claims["sub"], "role": role, "permissions": [role]}


# --- The server's auth handlers ---------------------------------------------------------------------------

auth = Auth()


@auth.authenticate
async def authenticate(authorization: str | None) -> dict:
    """Step 1, on every request: turn the `Authorization` header into a user, or refuse with 401."""
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Sign in first: send 'Authorization: Bearer <token>'.")
    try:
        return user_for(verify_token(token))
    except jwt.InvalidTokenError as err:
        raise Auth.exceptions.HTTPException(status_code=401, detail=f"Invalid token: {err}") from None


def is_staff(ctx: Auth.types.AuthContext) -> bool:
    return "staff" in ctx.permissions


def stamp_owner(ctx: Auth.types.AuthContext, value: dict) -> None:
    """Record who created a resource, in its metadata (mutate in place: the server keeps this dict)."""
    if value.get("metadata") is None:
        value["metadata"] = {}
    value["metadata"]["owner"] = ctx.user.identity


@auth.on
async def owner_only(ctx: Auth.types.AuthContext, value: dict) -> dict | None:
    """Default for threads, runs and crons: customers see and touch only what they created; staff see all."""
    if isinstance(value, dict) and ctx.action == "create":
        stamp_owner(ctx, value)
    return None if is_staff(ctx) else {"owner": ctx.user.identity}


@auth.on.assistants
async def assistants(ctx: Auth.types.AuthContext, value: dict) -> None:
    """Everyone may use the assistants; only staff may create, change or delete them."""
    if ctx.action not in ("read", "search") and not is_staff(ctx):
        raise Auth.exceptions.HTTPException(status_code=403, detail="Only staff can manage assistants.")


@auth.on.threads.create_run
async def run_as_caller(ctx: Auth.types.AuthContext, value: dict) -> dict | None:
    """Authorization can rewrite a request: a customer's runs always use *their* account as the context.

    The client may send any `context`; this replaces the customer fields, so `my_account` can't be pointed at
    someone else's account. (Staff runs keep the context they send.)
    """
    if is_staff(ctx):
        return None
    kwargs = value.setdefault("kwargs", {})
    account = ACCOUNTS[ctx.user.identity]
    context = kwargs.get("context") or {}
    context.update(account_id=ctx.user.identity, plan=account.plan)
    kwargs["context"] = context
    return {"owner": ctx.user.identity}


@auth.on.store()
async def own_memories(ctx: Auth.types.AuthContext, value: dict) -> None:
    """Store namespaces are `(kind, account_id, ...)` (Part 4): customers may only use their own account's."""
    namespace = tuple(value.get("namespace") or ())
    if not is_staff(ctx) and namespace[1:2] != (ctx.user.identity,):
        raise Auth.exceptions.HTTPException(status_code=403, detail="That memory belongs to another account.")
