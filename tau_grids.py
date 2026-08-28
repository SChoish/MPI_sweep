"""Default step-size grids used by MPI sweeps."""

from __future__ import annotations

MPI_TAU_GRID: tuple[float, ...] = (
    0.05,
    0.1,
    0.2,
    0.4,
    0.7,
    1.5,
    2.5,
    4.0,
    7.0,
    10.0,
    12.0,
    14.0,
    17.0,
    20.0,
)


def format_taus(values: tuple[float, ...], n: int | None = None) -> list[str]:
    """Return stable, filename-friendly representations of the first ``n`` values."""
    if n is not None and n < 1:
        raise ValueError("n must be positive or None")
    chosen = values if n is None else values[:n]
    return [f"{value:g}" for value in chosen]


def mpi_tau_grid(n: int | None = None) -> list[str]:
    """Return the default MPI sweep grid, optionally truncated to ``n`` values."""
    return format_taus(MPI_TAU_GRID, n)
