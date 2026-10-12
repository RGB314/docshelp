"""Auth: who is calling (authentication), and what they may see and do (authorization).

Academy: Introduction to LangSmith Deployment M5 "Introduction to Auth", "Authentication", "Authorization",
"Auth Provider", "Auth on Tutor".

Concepts (the handlers are in `docshelp/auth.py`; `langgraph.auth.json` adds `"auth": {"path": ...}`):
  * authentication: `@auth.authenticate` turns the `Authorization` header into a user (`identity`,
    `permissions`, anything else) or refuses with 401. Every request, every resource.
  * the auth provider: a login service issues signed tokens (JWTs); the server only *verifies* them. Here
    `issue_token` stands in for the provider (Supabase, Auth0, Clerk, Entra ID...): it signs with a local
    secret, so the lesson is free and offline. Real providers sign with a private key and publish the
    public keys (JWKS); `verify_token` is the only code that changes.
  * authorization: `@auth.on` handlers per resource and action (threads, runs, assistants, crons, store):
      - resource ownership: stamp `metadata["owner"]` on create and return the filter `{"owner": me}`,
        so customers never see each other's threads (a 404, not a 403: their existence isn't revealed)
      - roles: only staff may manage assistants (403 for customers); staff see every thread
      - rewriting requests: a customer's run always gets *their* account as context, whatever the client
        sends ("auth on the agent": `my_account` can't be pointed at someone else's account)
      - the Store: customers may only use namespaces of their own account
  * Studio still works locally: `langgraph dev` lets Studio in as a special user unless
    `"disable_studio_auth": true`. Custom routes are public unless `"enable_custom_route_auth": true`.

Use case: Ada (Acme) and Grace (Globex) use DocsHelp through the API; a support engineer oversees both.

Run:  uv run poe server-auth   (terminal 1: the same graphs with auth, on port 2025)
      uv run poe lesson 8.10   (terminal 2)
"""

import argparse

from langgraph_sdk.errors import AuthenticationError, NotFoundError, PermissionDeniedError

from docshelp.auth import issue_token
from docshelp.config import ensure_llm_ready
from docshelp.console import header
from docshelp.deployment import DEFAULT_AUTH_URL, connect, last_text, server_url


def as_user(url: str, token: str | None):
    headers = {"Authorization": f"Bearer {token}"} if token else None
    return connect(url, headers=headers, start="uv run poe server-auth")


def refused(call) -> str:
    """Run an API call that should fail; return how it failed."""
    try:
        call()
    except AuthenticationError:
        return "401 unauthenticated"
    except PermissionDeniedError:
        return "403 forbidden"
    except NotFoundError:
        return "404 not found"
    return "allowed"


def run(url: str) -> dict:
    results = {}
    header("Authentication")
    anonymous = as_user(url, None)
    results["no_token"] = refused(lambda: anonymous.threads.search())
    expired = as_user(url, issue_token("acct-1001", ttl_seconds=-60))
    results["expired"] = refused(lambda: expired.threads.search())
    print(f"No token: {results['no_token']}. Expired token: {results['expired']}.")
    ada = as_user(url, issue_token("acct-1001"))  # what Acme's login page would hand Ada's browser
    grace = as_user(url, issue_token("acct-2002"))
    staff = as_user(url, issue_token("support-engineer", role="staff"))
    print("Ada (acct-1001), Grace (acct-2002) and a support engineer are signed in.")

    header("Authorization: owners only")
    thread = ada.threads.create(metadata={"topic": "billing"})
    print(f"Ada's thread is stamped by the server: metadata={thread['metadata']}")
    results["grace_reads_adas_thread"] = refused(lambda: grace.threads.get(thread["thread_id"]))
    results["grace_sees"] = len(grace.threads.search(limit=100))
    results["staff_sees_adas_thread"] = any(
        t["thread_id"] == thread["thread_id"] for t in staff.threads.search(limit=100)
    )
    print(f"Grace opens it: {results['grace_reads_adas_thread']}. Grace's thread list: {results['grace_sees']} threads.")
    print(f"The support engineer can see it: {results['staff_sees_adas_thread']}")

    header("Auth on the agent: the run's context comes from the token")
    run_ = ada.runs.create(
        thread["thread_id"],
        "support_agent",
        input={"messages": [{"role": "user", "content": "Which company is my account for, and how many seats?"}]},
        context={"name": "Ada", "account_id": "acct-2002"},  # trying to read Globex's account
    )
    ada.runs.join(thread["thread_id"], run_["run_id"])
    used = ada.runs.get(thread["thread_id"], run_["run_id"])["kwargs"]["context"]
    results["context_account"] = used["account_id"]
    answer = last_text(ada.threads.get_state(thread["thread_id"])["values"])
    print(f"Ada asked as acct-2002; the server ran it with account_id={used['account_id']}.\nDocsHelp: {answer}")

    header("Roles and the Store")
    results["customer_creates_assistant"] = refused(lambda: ada.assistants.create("support_agent", name="mine"))
    assistant = staff.assistants.create("support_agent", name="staff-test", context={"style": "detailed"})
    staff.assistants.delete(assistant["assistant_id"])
    print(f"Ada creates an assistant: {results['customer_creates_assistant']}. Staff: allowed.")
    ada.store.put_item(["instructions", "acct-1001"], "instructions", {"text": "Short answers, please."})
    results["own_memory"] = "allowed"
    results["other_memory"] = refused(lambda: ada.store.get_item(["profile", "acct-2002"], "profile"))
    print(f"Ada writes her own memories: allowed. Reads Globex's: {results['other_memory']}.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default=server_url(DEFAULT_AUTH_URL, "LANGGRAPH_AUTH_URL"), help="the server with auth enabled"
    )
    args = parser.parse_args()
    ensure_llm_ready()
    run(args.url)


if __name__ == "__main__":
    main()
