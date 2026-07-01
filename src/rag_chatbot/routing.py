"""Shared routing prompt and response parsing."""

from typing import Literal

from pydantic import BaseModel, Field


Route = Literal["accounting", "general", "hr", "legal"]

ROUTER_SYSTEM_PROMPT = """You are a helpful assistant that classifies user queries into one of the following domains:
accounting, general, hr, legal.
- "legal": questions about laws, regulations, contracts, court cases, compliance
- "accounting": questions about finance, tax, IFRS, budgets, invoices, chart of accounts
- "hr": questions about employees, policies, onboarding, leave, benefits, conduct
- "general": questions about anything that does not clearly fit the above categories
Classify the user's question into exactly one domain.
Output a JSON object with the single key "query_type"."""


class QueryClassifier(BaseModel):
    query_type: Route = Field(
        ..., description="The accounting, general, HR, or legal destination"
    )
