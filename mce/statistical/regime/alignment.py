"""Canonicalize arbitrary latent IDs by interpretable training centroids."""

from mce.statistical.models import LatentStateProfile


def canonical_state_mapping(
    profiles: tuple[LatentStateProfile, ...], feature_names: tuple[str, ...]
) -> dict[int, int]:
    if not profiles or len({profile.state_id for profile in profiles}) != len(profiles):
        raise ValueError("profiles need unique state IDs")
    if any(len(profile.centroid) != len(feature_names) for profile in profiles):
        raise ValueError("profile centroids must match feature schema")
    keys = ("stress_score", "volatility_score", "market_direction_score")
    if any(key not in feature_names for key in keys):
        raise ValueError("alignment features missing from schema")
    indices = tuple(feature_names.index(key) for key in keys)

    def sort_key(profile: LatentStateProfile) -> tuple[float, float, float, int]:
        stress, volatility, direction = (profile.centroid[index] for index in indices)
        return (
            float("inf") if stress is None else stress,
            float("inf") if volatility is None else volatility,
            float("inf") if direction is None else -direction,
            profile.state_id,
        )

    return {
        profile.state_id: canonical
        for canonical, profile in enumerate(sorted(profiles, key=sort_key))
    }


def relabel_states(states: tuple[int, ...], mapping: dict[int, int]) -> tuple[int, ...]:
    if any(state not in mapping for state in states):
        raise ValueError("unmapped latent state")
    return tuple(mapping[state] for state in states)
