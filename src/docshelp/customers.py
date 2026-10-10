"""Who the agent is talking to: the runtime context type and a stand-in for the billing database.

`Customer` is the agent's *runtime context* (Part 1.9, Part 2.2): facts the app already knows about the
caller, passed per run with `agent.invoke(..., context=customer)`. Tools read it through `ToolRuntime`, so
the model never chooses whose data it sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Plan = Literal["free", "team", "enterprise"]


@dataclass
class Customer:
    """Runtime context: facts about the caller that the app knows (not the model). Passed per run."""

    name: str = "there"
    plan: Plan = "free"
    account_id: str = "acct-demo"


@dataclass(frozen=True)
class Account:
    company: str
    plan: Plan
    seats: int
    seats_used: int
    renews: str
    region: str
    contacts: tuple[str, ...]

    def describe(self) -> str:
        return (
            f"{self.company}: {self.plan} plan, {self.seats_used} of {self.seats} seats used, "
            f"renews {self.renews}, data region {self.region}."
        )


# A tiny stand-in for the billing system. Real apps would query a database or an internal API here.
ACCOUNTS: dict[str, Account] = {
    "acct-demo": Account("Demo workspace", "free", 1, 1, "never (free plan)", "us-east", ("demo@example.com",)),
    "acct-1001": Account("Acme Research", "team", 12, 11, "2027-01-31", "eu-west", ("ada@acme.example",)),
    "acct-2002": Account("Globex", "enterprise", 250, 231, "2027-06-30", "us-east", ("grace@globex.example",)),
}
