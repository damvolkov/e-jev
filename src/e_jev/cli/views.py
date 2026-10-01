"""cli.views: a System One response for the terminal — one row per answer, the distribution as bars. Pure rendering."""

from typing import Final

from rich.console import Console
from rich.table import Table
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer
from typesafe_sdk._core.response_types import ListModelsResponse, SystemOneResponse

HEADER: Final = "bold #6f9bff"
WIDTH: Final = 24
TOP: Final = 5
YES_AT: Final = 0.5


def bar_probability(probability: float) -> str:
    filled = round(probability * WIDTH)
    return f"[#6f9bff]{'█' * filled}[/][dim]{'░' * (WIDTH - filled)}[/] {probability:6.1%}"


def describe_answer(answer: NoulAnswer | ChoiceAnswer | ScoreAnswer) -> tuple[str, str, str, str]:
    """(kind, answer, confidence, distribution) for one answer; a noul's distribution is its yes and no."""
    match answer:
        case NoulAnswer(noul=noul):
            spread = {"yes": noul, "no": 1 - noul}
            return "noul", f"{'yes' if noul >= YES_AT else 'no'}  ({noul:.3f})", "—", render_spread(spread)
        case ChoiceAnswer(choice=choice, confidence=confidence, probabilities=probabilities):
            return "choice", choice, f"{confidence:.3f}", render_spread(probabilities)
        case ScoreAnswer(score=score, confidence=confidence, probabilities=probabilities, legend=legend):
            spread = {f"{level} {legend[level]}": value for level, value in probabilities.items()}
            return "score", f"{score:.2f}", f"{confidence:.3f}", render_spread(spread)


def render_spread(probabilities: dict) -> str:
    ranked = sorted(probabilities.items(), key=lambda item: -item[1])[:TOP]
    width = max(len(str(name)) for name, _ in ranked)
    rest = f"\n[dim]… {len(probabilities) - TOP} more[/]" if len(probabilities) > TOP else ""
    return "\n".join(f"{name!s:<{width}} {bar_probability(value)}" for name, value in ranked) + rest


def render_response(console: Console, response: SystemOneResponse, *, raw: bool) -> None:
    match raw:
        case True:
            console.print_json(response.model_dump_json())
        case False:
            table = Table("question", "type", "answer", "confidence", "distribution", box=None, header_style=HEADER, show_lines=True)
            for name, answer in response.answers.items():
                table.add_row(name, *describe_answer(answer))
            console.print(table)
            usage = response.usage
            console.print(f"[dim]{response.model} · {usage.input_tokens} in / {usage.output_tokens} out · request {response.request_id}[/]")


def render_models(console: Console, models: ListModelsResponse, *, raw: bool) -> None:
    match raw:
        case True:
            console.print_json(models.model_dump_json())
        case False:
            table = Table("model", "released", "description", box=None, header_style=HEADER)
            for model in models.models:
                table.add_row(model.name, model.release_date, model.description)
            console.print(table)
