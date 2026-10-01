"""examples.triage.graph: a support-ticket triage graph whose every judgement is a Jev question.

Pydantic AI turns each output model into System One questions — a `bool` is a noul, a str `Enum` a
choice (30 areas, past the 26 single-letter labels), an `IntEnum` with a docstring per level a score —
and pydantic-graph wires the steps: screen for spam, branch, classify, route. Point it at e-jev or at
TypeSafe's Jev with JEV_URL; the code does not change.

    uv run python -m examples.triage.graph "My card was charged twice this month"
"""

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, BoolCriteria, UseEnumMemberDocstrings
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider
from pydantic_graph import Graph, GraphBuilder, StepContext
from pydantic_settings import BaseSettings, SettingsConfigDict
from typesafe_sdk import AsyncTypeSafeClient

INSTRUCTIONS: Final = "You triage support tickets for a B2B analytics SaaS."


##### SETTINGS #####
class TriageSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JEV_")

    url: str = "http://localhost:45160"
    key: str = "local"
    model: str = "jev-latest"


##### MODELS #####
class Area(UseEnumMemberDocstrings, StrEnum):
    BILLING = "billing"
    """Charges, invoices, refunds, plans and payment methods."""
    ACCOUNT = "account"
    """Login, passwords, two-factor, user seats and roles."""
    API = "api"
    """The public REST API: errors, rate limits, keys, outages."""
    DASHBOARDS = "dashboards"
    """Building, sharing or viewing dashboards."""
    EXPORTS = "exports"
    """Exporting data or dashboards to CSV, PDF or spreadsheets."""
    INGESTION = "ingestion"
    """Data arriving late, missing or duplicated from connectors."""
    CONNECTORS = "connectors"
    """Setting up or authorizing a data source connector."""
    ALERTS = "alerts"
    """Threshold alerts and notifications."""
    SECURITY = "security"
    """Vulnerabilities, suspicious access, compliance reports."""
    PRIVACY = "privacy"
    """Personal data requests, GDPR, data deletion."""
    PERFORMANCE = "performance"
    """Slow queries or slow pages."""
    MOBILE = "mobile"
    """The iOS or Android app."""
    SSO = "sso"
    """SAML or OIDC single sign-on."""
    AUDIT = "audit"
    """Audit logs of who did what."""
    EMBEDDING = "embedding"
    """Embedding charts in another website."""
    SQL = "sql"
    """Writing or running SQL queries in the editor."""
    MODELS = "models"
    """Semantic models, metrics and dimensions."""
    SCHEDULING = "scheduling"
    """Scheduled reports and refresh times."""
    SHARING = "sharing"
    """Permissions on shared content."""
    LOCALIZATION = "localization"
    """Languages, time zones, currencies and number formats."""
    ACCESSIBILITY = "accessibility"
    """Screen readers, contrast, keyboard navigation."""
    BILLING_TAX = "billing_tax"
    """VAT numbers and tax on invoices."""
    CONTRACTS = "contracts"
    """Enterprise contracts, legal terms and procurement."""
    ONBOARDING = "onboarding"
    """Getting started, trials and first setup."""
    TRAINING = "training"
    """Courses, documentation requests and how-to questions."""
    PARTNERS = "partners"
    """Resellers and agency partnerships."""
    FEATURE = "feature"
    """Requests for a capability the product lacks."""
    FEEDBACK = "feedback"
    """Praise or general opinions with nothing to fix."""
    CANCELLATION = "cancellation"
    """Wanting to cancel or downgrade."""
    OTHER = "other"
    """Nothing above fits."""


class Priority(UseEnumMemberDocstrings, IntEnum):
    LOW = 0
    """A question or wish; nothing is broken."""
    NORMAL = 1
    """Something is wrong for one user, with a workaround."""
    HIGH = 2
    """A team is blocked or money is wrong."""
    CRITICAL = 3
    """Production is down or data is leaking, right now."""


class Screening(BaseModel):
    """Decide whether a support ticket deserves an agent's time."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    spam: Annotated[
        bool, BoolCriteria(true="Unsolicited advertising, scams or gibberish.", false="A real customer writing about the product.")
    ]
    """Is this ticket spam?"""


class Classification(BaseModel):
    """Classify a support ticket so it reaches the right team at the right speed."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    area: Area
    """Which part of the product is the ticket about?"""
    priority: Priority
    """How urgent is it?"""
    escalate: Annotated[
        bool, BoolCriteria(true="Legal, security or churn risk; a manager must see it.", false="The normal queue can handle it.")
    ]
    """Must a manager see this ticket today?"""


@dataclass(frozen=True, slots=True)
class Verdict:
    queue: str
    page: bool


@dataclass
class TriageState:
    """What the run learned on the way, kept for inspection: the ticket, each judgement, its confidence."""

    ticket: str = ""
    screening: Screening | None = None
    classification: Classification | None = None
    confidence: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TriageDeps:
    screen: Agent[None, Screening]
    classify: Agent[None, Classification]

    @classmethod
    @asynccontextmanager
    async def open(cls, settings: TriageSettings) -> AsyncIterator[Self]:
        """Both agents over one TypeSafe client, closed when the run ends."""
        async with AsyncTypeSafeClient(api_key=settings.key, base_url=settings.url) as client:
            model = TypeSafeModel(settings.model, provider=TypeSafeProvider(typesafe_client=client))
            yield cls(
                screen=Agent(model, output_type=Screening, instructions=INSTRUCTIONS),
                classify=Agent(model, output_type=Classification, instructions=INSTRUCTIONS),
            )


##### STEPS #####
async def screen(ctx: StepContext[TriageState, TriageDeps, str]) -> Screening:
    result = await ctx.deps.screen.run(ctx.inputs)
    ctx.state.ticket = ctx.inputs
    ctx.state.screening = result.output
    ctx.state.confidence |= (result.response.provider_details or {}).get("confidence", {})
    return result.output


async def discard(ctx: StepContext[TriageState, TriageDeps, Screening]) -> Verdict:
    return Verdict(queue="spam", page=False)


async def classify(ctx: StepContext[TriageState, TriageDeps, Screening]) -> Classification:
    result = await ctx.deps.classify.run(ctx.state.ticket)
    ctx.state.classification = result.output
    ctx.state.confidence |= (result.response.provider_details or {}).get("confidence", {})
    return result.output


async def route(ctx: StepContext[TriageState, TriageDeps, Classification]) -> Verdict:
    """Code decides what the judgements mean: the queue is the area, a page needs critical and escalation."""
    return Verdict(queue=ctx.inputs.area.value, page=ctx.inputs.priority is Priority.CRITICAL and ctx.inputs.escalate)


##### GRAPH #####
def build_triage() -> Graph[TriageState, TriageDeps, str, Verdict]:
    """Steps registered explicitly; the branch reads the screening, never the ticket."""
    g = GraphBuilder(state_type=TriageState, deps_type=TriageDeps, input_type=str, output_type=Verdict)
    screening, discarding, classifying, routing = g.step(screen), g.step(discard), g.step(classify), g.step(route)
    g.add(
        g.edge_from(g.start_node).to(screening),
        g.edge_from(screening).to(
            g.decision()
            .branch(g.match(Screening, matches=lambda result: result.spam).label("spam").to(discarding))
            .branch(g.match(Screening).label("customer").to(classifying))
        ),
        g.edge_from(classifying).to(routing),
        g.edge_from(discarding, routing).to(g.end_node),
    )
    return g.build()


async def triage(ticket: str, settings: TriageSettings) -> tuple[Verdict, TriageState]:
    state = TriageState()
    async with TriageDeps.open(settings) as deps:
        verdict = await build_triage().run(state=state, deps=deps, inputs=ticket)
    return verdict, state


def main() -> None:
    verdict, state = asyncio.run(triage(sys.argv[1], TriageSettings()))
    sys.stdout.write(f"{verdict}\n{state.classification}\n{state.confidence}\n\n{build_triage().render(title='triage')}\n")


if __name__ == "__main__":
    main()
