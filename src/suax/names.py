"""Resolve MLBAM player IDs to names.

Raw Statcast has a ``player_name`` column. It is the **pitcher**, not the
batter. Grouping by ``batter`` and taking the first ``player_name`` returns
whichever pitcher that hitter happened to face first -- which is how an early
version of this project produced a hitter leaderboard made entirely of
relievers. Always resolve from the ID.
"""

from __future__ import annotations

from typing import Dict, Iterable

__all__ = ["resolve_names"]


def resolve_names(ids: Iterable[int], verbose: bool = True) -> Dict[int, str]:
    """Map MLBAM IDs to display names via pybaseball. Returns {} on failure."""
    unique = sorted({int(i) for i in ids})
    if not unique:
        return {}
    try:
        from pybaseball import playerid_reverse_lookup
    except ImportError:
        if verbose:
            print("  pybaseball not installed - leaderboards will show IDs")
        return {}

    try:
        table = playerid_reverse_lookup(unique, key_type="mlbam")
    except Exception as exc:                      # network, API change, etc.
        if verbose:
            print(f"  name lookup failed ({exc}) - leaderboards will show IDs")
        return {}

    names = {
        int(row.key_mlbam): f"{str(row.name_first).title()} {str(row.name_last).title()}"
        for row in table.itertuples()
    }
    if verbose:
        print(f"  resolved {len(names):,} / {len(unique):,} names")
    return names
