from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse


MODEL_MESSAGES_GLOB = "*_ModelMessages.json"


@dataclass(frozen=True)
class BillableTokens:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class ResolvedTokenPrice:
    model_name: str
    prompt_usd_per_token: Decimal
    completion_usd_per_token: Decimal
    source: str


@dataclass
class PriceEstimate:
    prompt_usd_per_token: Decimal
    completion_usd_per_token: Decimal
    source: str
    input_usd: Decimal = Decimal("0")
    output_usd: Decimal = Decimal("0")
    total_usd: Decimal = Decimal("0")

    def add(self, billable: BillableTokens) -> None:
        input_cost = Decimal(billable.prompt_tokens) * self.prompt_usd_per_token
        output_cost = Decimal(billable.completion_tokens) * self.completion_usd_per_token
        self.input_usd += input_cost
        self.output_usd += output_cost
        self.total_usd += input_cost + output_cost


@dataclass
class UsageTotals:
    responses: int = 0
    missing_usage_responses: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    prompt_billable_tokens: int = 0
    completion_billable_tokens: int = 0

    def add_usage(self, usage: Any, billable: BillableTokens) -> None:
        self.input_tokens += _int_attr(usage, "input_tokens")
        self.output_tokens += _int_attr(usage, "output_tokens")
        self.reasoning_tokens += _detail_int(usage, "reasoning_tokens")
        self.prompt_billable_tokens += billable.prompt_tokens
        self.completion_billable_tokens += billable.completion_tokens

    def add_totals(self, other: "UsageTotals") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class ModelUsageSummary:
    model_name: str
    totals: UsageTotals = field(default_factory=UsageTotals)
    price: PriceEstimate | None = None
    price_unavailable_responses: int = 0

    def add_usage(
        self,
        usage: Any,
        billable: BillableTokens,
        resolved_price: ResolvedTokenPrice | None,
    ) -> None:
        self.totals.responses += 1
        self.totals.add_usage(usage, billable)
        if resolved_price is None:
            self.price_unavailable_responses += 1
            return
        if self.price is None:
            self.price = PriceEstimate(
                prompt_usd_per_token=resolved_price.prompt_usd_per_token,
                completion_usd_per_token=resolved_price.completion_usd_per_token,
                source=resolved_price.source,
            )
        elif self.price.source != resolved_price.source:
            sources = sorted({self.price.source, resolved_price.source})
            self.price.source = ", ".join(sources)
        self.price.add(billable)


@dataclass
class UsageFileSummary:
    path: Path
    meta: dict[str, Any]
    totals: UsageTotals = field(default_factory=UsageTotals)
    by_model: dict[str, ModelUsageSummary] = field(default_factory=dict)


@dataclass
class UsageReport:
    files: list[UsageFileSummary] = field(default_factory=list)
    totals: UsageTotals = field(default_factory=UsageTotals)
    by_model: dict[str, ModelUsageSummary] = field(default_factory=dict)


PriceResolver = Callable[[str], ResolvedTokenPrice | None]


def _int_attr(obj: Any, attr: str) -> int:
    value = getattr(obj, attr, 0)
    return value if isinstance(value, int) and value > 0 else 0


def _detail_int(usage: Any, key: str) -> int:
    details = getattr(usage, "details", None)
    if not isinstance(details, dict):
        return 0
    value = details.get(key)
    return value if isinstance(value, int) and value > 0 else 0


def _usage_has_values(usage: Any) -> bool:
    if usage is None:
        return False
    has_values = getattr(usage, "has_values", None)
    if callable(has_values):
        return bool(has_values())
    attrs = (
        "input_tokens",
        "cache_write_tokens",
        "cache_read_tokens",
        "output_tokens",
        "input_audio_tokens",
        "cache_audio_read_tokens",
        "output_audio_tokens",
    )
    return any(_int_attr(usage, attr) for attr in attrs) or any(
        _detail_int(usage, key)
        for key in (
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "reasoning_tokens",
        )
    )


def billable_tokens_from_usage(usage: Any) -> BillableTokens:
    prompt_cache_total = _detail_int(usage, "prompt_cache_hit_tokens") + _detail_int(
        usage, "prompt_cache_miss_tokens"
    )
    prompt_tokens = max(_int_attr(usage, "input_tokens"), prompt_cache_total)
    completion_tokens = _int_attr(usage, "output_tokens")

    return BillableTokens(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def read_model_messages_file(path: Path) -> tuple[list[Any], dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("model_messages")
    if not isinstance(raw, list):
        raise ValueError(f"invalid model_messages file: {path}")
    meta_raw = data.get("meta")
    meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}
    meta["saved_at"] = data.get("saved_at")
    return ModelMessagesTypeAdapter.validate_python(raw), meta


def model_message_files_for_path(path: Path) -> list[Path]:
    p = Path(path).expanduser()
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.rglob(MODEL_MESSAGES_GLOB), key=lambda item: str(item))
    return []


def session_model_message_files(conversations_root_path: Path, session_key: str) -> list[Path]:
    parts = str(session_key or "").split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return []
    date, topic = parts
    root = Path(conversations_root_path)
    if not root.is_dir():
        return []
    out: list[Path] = []
    for fp in root.glob(MODEL_MESSAGES_GLOB):
        try:
            with fp.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        meta_raw = data.get("meta")
        meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}
        if str(meta.get("date") or "") == date and str(meta.get("topic") or "") == topic:
            out.append(fp)
    return sorted(out, key=lambda item: str(item))


def latest_usage_input_tokens(messages: Iterable[Any]) -> int | None:
    for message in reversed(list(messages)):
        if isinstance(message, ModelResponse):
            usage = getattr(message, "usage", None)
            if _usage_has_values(usage):
                return _int_attr(usage, "input_tokens")
    return None


def summarize_messages(
    messages: Iterable[Any],
    *,
    meta: dict[str, Any] | None = None,
    path: Path | None = None,
    price_resolver: PriceResolver | None = None,
) -> UsageFileSummary:
    resolver = price_resolver or resolve_token_price
    summary = UsageFileSummary(path=Path(path) if path is not None else Path(), meta=meta or {})
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        summary.totals.responses += 1
        usage = getattr(message, "usage", None)
        if not _usage_has_values(usage):
            summary.totals.missing_usage_responses += 1
            continue
        model_name = str(getattr(message, "model_name", "") or "unknown")
        billable = billable_tokens_from_usage(usage)
        summary.totals.add_usage(usage, billable)
        model_summary = summary.by_model.setdefault(
            model_name, ModelUsageSummary(model_name=model_name)
        )
        model_summary.add_usage(usage, billable, resolver(model_name))
    return summary


def summarize_usage_files(
    paths: Iterable[Path],
    *,
    price_resolver: PriceResolver | None = None,
) -> UsageReport:
    report = UsageReport()
    for path in paths:
        messages, meta = read_model_messages_file(Path(path))
        file_summary = summarize_messages(
            messages,
            meta=meta,
            path=Path(path),
            price_resolver=price_resolver,
        )
        report.files.append(file_summary)
        report.totals.add_totals(file_summary.totals)
        for model_name, model_summary in file_summary.by_model.items():
            total_model = report.by_model.setdefault(
                model_name, ModelUsageSummary(model_name=model_name)
            )
            total_model.totals.add_totals(model_summary.totals)
            total_model.price_unavailable_responses += model_summary.price_unavailable_responses
            if model_summary.price is not None:
                if total_model.price is None:
                    total_model.price = PriceEstimate(
                        prompt_usd_per_token=model_summary.price.prompt_usd_per_token,
                        completion_usd_per_token=model_summary.price.completion_usd_per_token,
                        source=model_summary.price.source,
                        input_usd=model_summary.price.input_usd,
                        output_usd=model_summary.price.output_usd,
                        total_usd=model_summary.price.total_usd,
                    )
                else:
                    total_model.price.input_usd += model_summary.price.input_usd
                    total_model.price.output_usd += model_summary.price.output_usd
                    total_model.price.total_usd += model_summary.price.total_usd
                    if total_model.price.source != model_summary.price.source:
                        sources = sorted({total_model.price.source, model_summary.price.source})
                        total_model.price.source = ", ".join(sources)
    return report


def resolve_token_price(model_name: str) -> ResolvedTokenPrice | None:
    candidates = _openrouter_price_candidates(model_name) + _genai_price_candidates(model_name)
    prompt = _max_price(candidates, "prompt")
    completion = _max_price(candidates, "completion")
    if prompt is None or completion is None:
        return None
    source = ", ".join(sorted({prompt[1], completion[1]}))
    return ResolvedTokenPrice(
        model_name=model_name,
        prompt_usd_per_token=prompt[0],
        completion_usd_per_token=completion[0],
        source=source,
    )


def _max_price(
    candidates: list[tuple[Decimal | None, Decimal | None, str]],
    kind: str,
) -> tuple[Decimal, str] | None:
    index = 0 if kind == "prompt" else 1
    values: list[tuple[Decimal, str]] = []
    for candidate in candidates:
        value = candidate[index]
        if value is not None:
            values.append((value, candidate[2]))
    if not values:
        return None
    return max(values, key=lambda item: item[0])


def _openrouter_price_candidates(model_name: str) -> list[tuple[Decimal | None, Decimal | None, str]]:
    from ModelGateway.ModelChecker import _lookup_openrouter_meta

    try:
        meta = _lookup_openrouter_meta(model_name)
    except Exception:
        meta = None
    pricing = meta.get("pricing") if isinstance(meta, dict) else None
    if not isinstance(pricing, dict):
        return []
    prompt = _decimal_or_none(pricing.get("prompt"))
    completion = _decimal_or_none(pricing.get("completion"))
    if prompt is None and completion is None:
        return []
    return [(prompt, completion, "openrouter")]


def _genai_price_candidates(model_name: str) -> list[tuple[Decimal | None, Decimal | None, str]]:
    try:
        from genai_prices.data_snapshot import get_snapshot
    except Exception:
        return []
    try:
        snap = get_snapshot()
    except Exception:
        return []
    now = datetime.now(timezone.utc)
    out: list[tuple[Decimal | None, Decimal | None, str]] = []
    names = {model_name, model_name.lower()}
    if "/" in model_name:
        names.add(model_name.rsplit("/", 1)[-1])
    for provider in snap.providers:
        for model in provider.models:
            try:
                matched = any(model.is_match(name) for name in names)
            except Exception:
                matched = False
            if not matched:
                continue
            try:
                prices = model.get_prices(now)
            except Exception:
                prices = getattr(model, "prices", None)
            prompt = _mtok_to_token_price(getattr(prices, "input_mtok", None))
            completion = _mtok_to_token_price(getattr(prices, "output_mtok", None))
            if prompt is not None or completion is not None:
                provider_name = str(getattr(provider, "name", "") or getattr(provider, "id", ""))
                out.append((prompt, completion, f"genai-prices:{provider_name}"))
    return out


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _mtok_to_token_price(value: Any) -> Decimal | None:
    price = _max_mtok_price(value)
    if price is None:
        return None
    return price / Decimal(1_000_000)


def _max_mtok_price(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    base = getattr(value, "base", None)
    prices: list[Decimal] = []
    if isinstance(base, Decimal):
        prices.append(base)
    tiers = getattr(value, "tiers", None)
    if isinstance(tiers, list):
        for tier in tiers:
            tier_price = getattr(tier, "price", None)
            if isinstance(tier_price, Decimal):
                prices.append(tier_price)
    if not prices:
        return None
    return max(prices)
