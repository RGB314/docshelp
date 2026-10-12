---
name: refund-policy
description: Decide whether a Nimbus Notes customer gets a refund or credit, and how much. Use for any billing complaint, double charge, unused seats, or refund request.
---

# Refund policy

Use this when a customer asks for money back or says they were charged wrongly.

## Rules

1. **Billing errors** (double charge, charged for more seats than members): refund the difference in full,
   always. No approval needed.
2. **Unused seats** on the Team plan: credit the unused seats from the next billing date. No cash refund.
3. **Annual plans** cancelled within 14 days of purchase or renewal: full refund. After 14 days: no refund,
   the plan runs to the end of the term.
4. **Outages**: if an open incident blocked the customer for more than 24 hours, offer one month's credit.
5. Anything else, or more than $500: open a ticket with priority `high` for the billing team.

## How to answer

- Check the plan and seats first (`my_account`, or the ticket's account id).
- Quote the price with `quote_price` rather than doing arithmetic yourself. Team is $8 per member per month.
- State which rule applies by number, the amount, and when the customer will see it.
- Never promise more than the rules allow.
