"""Point-in-time member liquidity, actual quote spread, and gap aggregation."""

from dataclasses import dataclass
from datetime import date
from math import isfinite
from statistics import median

from mce.calendar.service import MarketSessionService
from mce.config.settings import LiquiditySettings, MCEConfig, StressSettings
from mce.features.breadth.membership import UniverseSnapshot
from mce.features.dispersion.cross_section import cross_section
from mce.features.stress.composite import dollar_volume_concentration, liquidity_score
from mce.features.stress.history import LiquidityHistory, prior_rank
from mce.features.stress.models import LiquidityContext, LiquidityMember
from mce.features.stress.primitives import read_session, read_spread
from mce.models.enums import DataQuality, QualityFlag
from mce.replay.store import CausalDataStore


@dataclass(frozen=True)
class GapEvidence:
    up_fraction: float | None
    down_fraction: float | None
    magnitude: float | None
    negative_magnitude: float | None
    dispersion: float | None
    percentile: float | None
    coverage: float
    stressed_member_fraction: float | None


def _finite_median(values: list[float]) -> float | None:
    value = median(values) if values else None
    return value if value is not None and isfinite(value) else None


def _membership_factor(quality: DataQuality) -> float:
    return {
        DataQuality.GOOD: 1.0,
        DataQuality.DEGRADED: 0.5,
        DataQuality.POOR: 0.25,
        DataQuality.UNUSABLE: 0.0,
        DataQuality.UNKNOWN: 0.5,
    }[quality]


def build_liquidity(
    store: CausalDataStore,
    calendar: MarketSessionService,
    universe: UniverseSnapshot,
    day: date | None,
    as_of_ns: int,
    history: LiquidityHistory,
    config: MCEConfig,
) -> tuple[LiquidityContext, GapEvidence]:
    settings: LiquiditySettings = config.features.liquidity
    stress_settings: StressSettings = config.features.stress
    expected = len(universe.members)
    flags = set(universe.quality_flags)
    if (
        universe.source_quality != DataQuality.GOOD
        or settings.price_adjustment_status != "adjusted"
    ):
        flags.add(QualityFlag.PROVIDER_DEGRADED)
    per_member: dict[str, LiquidityMember] = {}
    valid = stale_bars = stale_quotes = valid_quotes = 0
    volumes: list[float] = []
    dollars: list[float] = []
    rvols: list[float] = []
    rvols_5d: list[float] = []
    rvols_20d: list[float] = []
    amihuds: list[float] = []
    quoted: list[float] = []
    spreads: list[float] = []
    gaps: list[float] = []
    stressed = comparable_stress = 0
    large_move_weak = large_move_high = comparable_move_volume = 0
    if day is not None:
        opened, closed = calendar.regular_session_bounds(day)
    for member in universe.members:
        panel = (
            read_session(
                store,
                calendar,
                member.symbol,
                day,
                as_of_ns,
                config.provider.source,
                config.freshness.bar_limit_ns,
                settings.bar_cadence_ns,
            )
            if day is not None and settings.enabled
            else None
        )
        quote = (
            read_spread(
                store,
                member.symbol,
                as_of_ns,
                opened,
                closed,
                config.provider.source,
                config.freshness.quote_limit_ns,
            )
            if day is not None and settings.enabled and settings.quote_spread_enabled
            else None
        )
        volume = panel.volume if panel else None
        dollar = panel.dollar_volume if panel else None
        amihud = panel.amihud if panel and settings.amihud_enabled else None
        gap = panel.gap_return if panel else None
        if panel and panel.stale:
            stale_bars += 1
        if quote and quote.stale:
            stale_quotes += 1
        if quote and quote.invalid:
            flags.add(QualityFlag.REJECTED_DATA)
        if volume is not None:
            valid += 1
            volumes.append(volume)
        if dollar is not None:
            dollars.append(dollar)
        if amihud is not None:
            amihuds.append(amihud)
        if gap is not None:
            gaps.append(gap)
        if quote and quote.relative is not None and quote.absolute is not None:
            valid_quotes += 1
            quoted.append(quote.absolute)
            spreads.append(quote.relative)
        past_volume = history.member_volume.get(member.symbol, ())
        reference_volume = (
            _finite_median(list(past_volume))
            if len(past_volume) >= settings.minimum_history_samples
            else None
        )
        rvol = (
            volume / reference_volume
            if volume is not None and reference_volume and reference_volume > 0
            else None
        )
        if rvol is not None and isfinite(rvol):
            rvols.append(rvol)
        else:
            rvol = None
        if panel and panel.session_return is not None and rvol is not None:
            comparable_move_volume += 1
            if abs(panel.session_return) >= settings.large_move_threshold:
                large_move_weak += rvol < settings.weak_volume_rvol_threshold
                large_move_high += rvol > settings.high_volume_rvol_threshold
        for window, output in ((5, rvols_5d), (20, rvols_20d)):
            if volume is not None and len(past_volume) >= window:
                baseline = _finite_median(list(past_volume[:window]))
                if baseline is not None and baseline > 0:
                    ratio = volume / baseline
                    if isfinite(ratio):
                        output.append(ratio)
        past_amihud = history.member_illiquidity.get(member.symbol, ())
        illiq_rank = prior_rank(amihud, past_amihud, settings.minimum_history_samples)
        illiq_pct = illiq_rank.percentile
        gap_stressed = gap is not None and gap < -stress_settings.gap_scale
        if illiq_pct is not None or gap is not None:
            comparable_stress += 1
            if (
                illiq_pct is not None and illiq_pct >= stress_settings.stress_breadth_threshold
            ) or gap_stressed:
                stressed += 1
        per_member[member.symbol] = LiquidityMember(
            symbol=member.symbol,
            weight=member.weight,
            session_volume=volume,
            session_dollar_volume=dollar,
            session_return=panel.session_return if panel else None,
            relative_volume=rvol,
            illiquidity_proxy=amihud,
            illiquidity_percentile=illiq_pct,
            quoted_spread=quote.absolute if quote else None,
            relative_quoted_spread=quote.relative if quote else None,
            gap_return=gap,
            stale_bar=panel.stale if panel else False,
            stale_quote=quote.stale if quote else False,
            explicit_zero_volume=panel.zero_volume if panel else False,
        )
    coverage = valid / expected if expected else 0.0
    quote_coverage = valid_quotes / expected if expected else 0.0
    if coverage < 1:
        flags.add(QualityFlag.PARTIAL_UNIVERSE)
    if coverage < settings.minimum_member_coverage:
        flags.add(QualityFlag.LOW_COVERAGE)
    if stale_bars or stale_quotes:
        flags.add(QualityFlag.STALE_DATA)
    if day is None:
        flags.add(QualityFlag.OUTSIDE_SESSION)
    if settings.quote_spread_enabled and quote_coverage < settings.minimum_quote_coverage:
        flags.add(QualityFlag.MISSING_DATA)
    total_volume = sum(volumes) if volumes else None
    total_dollar = sum(dollars) if dollars else None
    total_volume = total_volume if total_volume is not None and isfinite(total_volume) else None
    total_dollar = total_dollar if total_dollar is not None and isfinite(total_dollar) else None
    volume_rank = prior_rank(total_volume, history.market_volume, settings.minimum_history_samples)
    illiquidity = _finite_median(amihuds)
    illiq_rank = prior_rank(
        illiquidity, history.market_illiquidity, settings.minimum_history_samples
    )
    if volume_rank.percentile is None or (
        settings.amihud_enabled and illiq_rank.percentile is None
    ):
        flags.add(QualityFlag.INSUFFICIENT_HISTORY)
    enough_members = (
        valid >= settings.minimum_members and coverage >= settings.minimum_member_coverage
    )
    enough_rvol = (
        len(rvols) >= settings.minimum_members
        and len(rvols) / expected >= settings.minimum_member_coverage
        if expected
        else False
    )
    relative_volume = _finite_median(rvols) if enough_rvol else None
    relative_volume_5d = (
        _finite_median(rvols_5d)
        if len(rvols_5d) >= settings.minimum_members
        and expected
        and len(rvols_5d) / expected >= settings.minimum_member_coverage
        else None
    )
    relative_volume_20d = (
        _finite_median(rvols_20d)
        if len(rvols_20d) >= settings.minimum_members
        and expected
        and len(rvols_20d) / expected >= settings.minimum_member_coverage
        else None
    )
    move_volume_coverage = comparable_move_volume / expected if expected else 0.0
    enough_move_volume = (
        comparable_move_volume >= settings.minimum_members
        and move_volume_coverage >= settings.minimum_member_coverage
    )
    spread_ok = (
        valid_quotes >= settings.minimum_members
        and quote_coverage >= settings.minimum_quote_coverage
    )
    relative_spread = _finite_median(spreads) if spread_ok else None
    absolute_spread = _finite_median(quoted) if spread_ok else None
    if not enough_members:
        relative_volume = None
    score, active_weights, contributors = liquidity_score(
        illiq_rank.percentile if enough_members else None,
        relative_volume,
        relative_spread,
        settings,
    )
    all_weighted = bool(universe.members) and all(
        member.weight is not None for member in universe.members
    )
    weight_coverage = weighted_illiquidity = None
    if all_weighted:
        total_weight = sum(member.weight or 0 for member in universe.members)
        selected = tuple(
            (per_member[member.symbol].illiquidity_proxy, member.weight or 0)
            for member in universe.members
            if per_member[member.symbol].illiquidity_proxy is not None
        )
        weight_coverage = (
            sum(weight for _, weight in selected) / total_weight
            if total_weight > 0 and isfinite(total_weight)
            else None
        )
        if weight_coverage is not None and not isfinite(weight_coverage):
            weight_coverage = None
        if weight_coverage is not None and weight_coverage >= settings.minimum_weight_coverage:
            selected_weight = sum(weight for _, weight in selected)
            weighted_illiquidity = (
                sum(
                    value * (weight / selected_weight)
                    for value, weight in selected
                    if value is not None
                )
                if selected_weight > 0
                else None
            )
            if weighted_illiquidity is not None and not isfinite(weighted_illiquidity):
                weighted_illiquidity = None
    elif any(member.weight is not None for member in universe.members):
        flags.add(QualityFlag.MISSING_DATA)
    illi_stats = cross_section(tuple(amihuds))
    rvol_stats = cross_section(tuple(rvols))
    spread_stats = cross_section(tuple(spreads))
    concentration = dollar_volume_concentration(tuple(dollars))
    change = None
    if illiquidity is not None and illiq_rank.median is not None:
        denominator = illiquidity + illiq_rank.median
        change = (illiq_rank.median - illiquidity) / denominator if denominator > 0 else 0.0
    if score is None:
        flags.add(QualityFlag.INSUFFICIENT_HISTORY)
    adjustment = 1.0 if settings.price_adjustment_status == "adjusted" else 0.5
    depth = min(1.0, volume_rank.sample_count / settings.minimum_history_samples)
    evidence_quality = (
        coverage
        * _membership_factor(universe.source_quality)
        * adjustment
        * (0.5 + 0.25 * depth + 0.25 * quote_coverage)
    )
    quality = (
        DataQuality.UNUSABLE
        if not enough_members or universe.source_quality == DataQuality.UNUSABLE
        else DataQuality.POOR
        if score is None
        else DataQuality.DEGRADED
        if flags
        else DataQuality.GOOD
    )
    gap_count = len(gaps)
    gap_coverage = gap_count / expected if expected else 0.0
    gap_rank = prior_rank(
        _finite_median([abs(value) for value in gaps]),
        history.market_gap_magnitude,
        settings.minimum_history_samples,
    )
    negative = tuple(-value for value in gaps if value < 0)
    gap_stats = cross_section(tuple(gaps))
    gap_evidence = GapEvidence(
        up_fraction=sum(value > 0 for value in gaps) / gap_count if gap_count else None,
        down_fraction=len(negative) / gap_count if gap_count else None,
        magnitude=_finite_median([abs(value) for value in gaps]),
        negative_magnitude=_finite_median(list(negative))
        if negative
        else 0.0
        if gap_count
        else None,
        dispersion=gap_stats.std if gap_stats else None,
        percentile=gap_rank.percentile,
        coverage=gap_coverage,
        stressed_member_fraction=stressed / comparable_stress
        if comparable_stress >= settings.minimum_members
        else None,
    )
    liquidity_context = LiquidityContext(
        per_member=per_member,
        expected_member_count=expected,
        valid_member_count=valid,
        stale_bar_count=stale_bars,
        stale_quote_count=stale_quotes,
        member_coverage=coverage,
        quote_coverage=quote_coverage,
        weight_coverage=weight_coverage,
        session_volume=total_volume,
        session_dollar_volume=total_dollar,
        relative_volume=relative_volume,
        relative_volume_5d=relative_volume_5d,
        relative_volume_20d=relative_volume_20d,
        relative_volume_dispersion=rvol_stats.std if rvol_stats else None,
        volume_percentile=volume_rank.percentile,
        volume_robust_z=volume_rank.robust_z,
        volume_breadth=sum(value > 1 for value in rvols) / len(rvols) if rvols else None,
        large_move_weak_volume_fraction=large_move_weak / comparable_move_volume
        if enough_move_volume
        else None,
        large_move_high_volume_fraction=large_move_high / comparable_move_volume
        if enough_move_volume
        else None,
        illiquidity_proxy=illiquidity,
        weighted_illiquidity_proxy=weighted_illiquidity,
        illiquidity_dispersion=illi_stats.std if illi_stats else None,
        illiquidity_percentile=illiq_rank.percentile,
        illiquidity_robust_z=illiq_rank.robust_z,
        price_impact_proxy=illiquidity,
        quoted_spread=absolute_spread,
        relative_quoted_spread=relative_spread,
        spread_dispersion=spread_stats.std if spread_stats else None,
        dollar_volume_hhi=concentration[0] if concentration else None,
        effective_active_names=concentration[1] if concentration else None,
        liquidity_change=change,
        liquidity_score=score,
        active_weights=active_weights,
        evidence_quality=evidence_quality,
        data_quality=quality,
        contributors=contributors,
        quality_flags=frozenset(flags),
    )
    return liquidity_context, gap_evidence
