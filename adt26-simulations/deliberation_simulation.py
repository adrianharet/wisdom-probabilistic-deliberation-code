from __future__ import annotations

import random
from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "DEFAULT_PLOT_PALETTE",
    "AgentState",
    "RoundState",
    "generate_evidence",
    "opinion_from_evidence",
    "collective_opinion",
    "Deliberation",
    "Batch_Deliberation",
    "plot_avg_fraction_correct_agents_at_termination_over_n_per_k",
    "plot_avg_fraction_correct_agents_over_rounds_per_k",
    "plot_avg_fraction_correct_agents_over_rounds_per_n",
    "plot_fraction_correct_decisions_at_termination_over_n_per_k",
    "plot_fraction_correct_decisions_over_rounds_per_k",
    "plot_fraction_correct_decisions_over_rounds_per_n",
    "plot_fraction_correct_decisions_over_n_per_round",
    "plot_fraction_correct_decisions_over_k_per_round",
    "plot_avg_number_of_rounds_over_n_per_k",
    "plot_avg_number_of_rounds_over_k_per_n",
    "plot_avg_number_of_successes_over_s_per_k",
    "plot_fraction_correct_decisions_at_termination_over_s",
    "plot_avg_number_of_rounds_over_s",
    "plot_avg_coverage_diversity_over_s",
    "plot_avg_pairwise_diversity_over_s",
    "batch_terminal_statistics",
    "deliberation",
    "batch_deliberation",
]

DEFAULT_PLOT_PALETTE = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51"]
ALTERNATIVE_COLORS = {"a": "#046C41", "b": "#8d0220", "tie": "#748186"}
FIGSIZE = (6, 4)
DEFAULT_PLOT_DIR = Path("simulations/plots")
DPI = 400

Opinion = str
EvidenceItem = str

@dataclass(frozen=True)
class AgentState:
    """Snapshot of one agent's evidence and current opinion at a given round."""

    evidence: tuple[EvidenceItem, ...]
    opinion: Opinion

@dataclass(frozen=True)
class RoundState:
    """Complete public record of a deliberation round."""

    round_index: int
    agents: tuple[AgentState, ...]
    collective_opinion: Opinion
    public_evidence: tuple[EvidenceItem, ...]
    disclosed_this_round: tuple[EvidenceItem, ...]
    disclosure: bool

def generate_evidence(m: int, l: int) -> tuple[EvidenceItem, ...]:
    """Build the full evidence pool with ``l`` items for 'a' and ``m - l`` for 'b'."""

    _validate_evidence_parameters(l=l, m=m)
    return tuple([f"a{i + 1}" for i in range(l)] + [f"b{i + 1}" for i in range(m - l)])

def opinion_from_evidence(evidence: Iterable[EvidenceItem]) -> Opinion:
    """Return the opinion supported by the balance of evidence in ``evidence``."""

    count_a = sum(1 for item in evidence if item.startswith("a"))
    count_b = sum(1 for item in evidence if item.startswith("b"))
    if count_a > count_b:
        return "a"
    if count_b > count_a:
        return "b"
    return "tie"

def collective_opinion(opinions: Sequence[Opinion]) -> Opinion:
    """Compute the strict majority opinion across agents, with ties returning ``'tie'``."""

    votes_for_a = sum(1 for opinion in opinions if opinion == "a")
    votes_for_b = sum(1 for opinion in opinions if opinion == "b")
    if votes_for_a > votes_for_b:
        return "a"
    if votes_for_b > votes_for_a:
        return "b"
    return "tie"

def Deliberation(
    k: int,
    l: int,
    m: int,
    n: int,
    *,
    s: float | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> list[RoundState]:
    """Run one deliberation instance from initial sampling through endogenous termination."""

    _validate_parameters(k=k, l=l, m=m, n=n)
    salience = _resolve_salience(s=s, l=l, m=m)
    _validate_salience_for_sampling(salience, k=k, l=l, m=m)
    generator = _resolve_rng(rng=rng, seed=seed)
    evidence_pool = generate_evidence(m=m, l=l)
    probabilities = _evidence_saliences(evidence_pool, s=salience)

    # Round 0: agents sample private evidence independently and form initial opinions.
    agents_evidence = [
        tuple(sorted(_sample_private_evidence(generator, evidence_pool, k, probabilities)))
        for _ in range(n)
    ]
    initial_agents = tuple(
        AgentState(evidence=evidence, opinion=opinion_from_evidence(evidence))
        for evidence in agents_evidence
    )
    history = [
        RoundState(
            round_index=0,
            agents=initial_agents,
            collective_opinion=collective_opinion([agent.opinion for agent in initial_agents]),
            public_evidence=tuple(),
            disclosed_this_round=tuple(),
            disclosure=True,
        )
    ]

    while True:
        previous_round = history[-1]
        previous_majority = previous_round.collective_opinion
        public_evidence = set(previous_round.public_evidence)
        disclosed_this_round: set[EvidenceItem] = set()

        # Disclosure is synchronous: every dissenter decides what to reveal from the
        # previous state before anyone updates on newly public evidence.
        for agent in previous_round.agents:
            if _disagrees_with_majority(agent.opinion, previous_majority):
                supported_evidence = {
                    item
                    for item in agent.evidence
                    if item.startswith(agent.opinion)
                }
                disclosed_this_round.update(supported_evidence - public_evidence)

        updated_public_evidence = public_evidence | disclosed_this_round
        next_agents = []
        # After disclosure is collected, every agent updates on the same public evidence.
        for agent in previous_round.agents:
            updated_evidence = tuple(sorted(set(agent.evidence) | updated_public_evidence))
            next_agents.append(
                AgentState(
                    evidence=updated_evidence,
                    opinion=opinion_from_evidence(updated_evidence),
                )
            )

        history.append(
            RoundState(
                round_index=previous_round.round_index + 1,
                agents=tuple(next_agents),
                collective_opinion=collective_opinion([agent.opinion for agent in next_agents]),
                public_evidence=tuple(sorted(updated_public_evidence)),
                disclosed_this_round=tuple(sorted(disclosed_this_round)),
                disclosure=bool(disclosed_this_round),
            )
        )

        if not disclosed_this_round:
            break

    return history

def Batch_Deliberation(
    k: int,
    l: int,
    m: int,
    n: int,
    batch_size: int,
    *,
    s: float | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> list[list[RoundState]]:
    """Run ``batch_size`` independent deliberation instances and collect their histories."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    generator = _resolve_rng(rng=rng, seed=seed)
    return [
        Deliberation(k=k, l=l, m=m, n=n, s=s, rng=generator)
        for _ in range(batch_size)
    ]

def batch_terminal_statistics(
    k: int,
    l: int,
    m: int,
    n: int,
    batch_size: int,
    *,
    s: float | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Return batch histories together with terminal fractions, mean, and standard deviation."""

    histories = Batch_Deliberation(
        k=k,
        l=l,
        m=m,
        n=n,
        batch_size=batch_size,
        s=s,
        rng=rng,
        seed=seed,
    )
    terminal_fractions = [_fraction_correct_at_termination(history) for history in histories]
    return {
        "histories": histories,
        "terminal_fractions": terminal_fractions,
        "mean_terminal_fraction": mean(terminal_fractions),
        "std_terminal_fraction": pstdev(terminal_fractions),
    }

def deliberation(
    k: int,
    l: int,
    m: int,
    n: int,
    *,
    s: float | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> list[RoundState]:
    """Lowercase alias for ``Deliberation``."""

    return Deliberation(k=k, l=l, m=m, n=n, s=s, rng=rng, seed=seed)

def batch_deliberation(
    k: int,
    l: int,
    m: int,
    n: int,
    batch_size: int,
    *,
    s: float | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> list[list[RoundState]]:
    """Lowercase alias for ``Batch_Deliberation``."""

    return Batch_Deliberation(
        k=k,
        l=l,
        m=m,
        n=n,
        batch_size=batch_size,
        s=s,
        rng=rng,
        seed=seed,
    )

def _fraction_correct_at_termination(history: Sequence[RoundState]) -> float:
    """Compute the terminal fraction of agents whose final opinion is 'a'."""

    final_round = history[-1]
    correct_agents = sum(1 for agent in final_round.agents if agent.opinion == "a")
    return correct_agents / len(final_round.agents)

def _correct_collective_decision_at_termination(history: Sequence[RoundState]) -> float:
    """Return 1.0 iff the terminal collective opinion is 'a'; otherwise return 0.0."""

    return 1.0 if history[-1].collective_opinion == "a" else 0.0

def _disagrees_with_majority(opinion: Opinion, majority: Opinion) -> bool:
    """Return whether an agent's non-tie opinion is the opposite of the previous majority."""

    return (opinion == "a" and majority == "b") or (opinion == "b" and majority == "a")

def _resolve_rng(
    *,
    rng: random.Random | None,
    seed: int | None,
) -> random.Random:
    """Normalize the random number source so the public API supports ``rng`` or ``seed``."""

    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both.")
    if rng is not None:
        return rng
    return random.Random(seed)

def _resolve_salience(s: float | None, l: int, m: int) -> float:
    """Return the active salience value, defaulting to the uniform baseline."""

    salience = l / m if s is None else s
    if not 0 <= salience <= 1:
        raise ValueError("s must satisfy 0 <= s <= 1.")
    return salience

def _validate_salience_for_sampling(s: float, k: int, l: int, m: int) -> None:
    """Validate that the salience distribution can supply ``k`` nonzero-probability items."""

    b_count = m - l
    if l == 0 and s > 0:
        raise ValueError("s must be 0 when there are no a-evidence items.")
    if b_count == 0 and s < 1:
        raise ValueError("s must be 1 when there are no b-evidence items.")
    if s == 1 and k > l:
        raise ValueError("Cannot sample k items from only a-evidence when s=1.")
    if s == 0 and k > b_count:
        raise ValueError("Cannot sample k items from only b-evidence when s=0.")

def _evidence_saliences(evidence_pool: Sequence[EvidenceItem], s: float) -> np.ndarray:
    """Convert total a-salience ``s`` into a per-item probability vector."""

    count_a = sum(1 for item in evidence_pool if item.startswith("a"))
    count_b = len(evidence_pool) - count_a
    probabilities = [
        s / count_a if item.startswith("a") else (1 - s) / count_b
        for item in evidence_pool
    ]
    return np.array(probabilities, dtype=float)

def _sample_private_evidence(
    generator: random.Random,
    evidence_pool: Sequence[EvidenceItem],
    k: int,
    probabilities: np.ndarray,
) -> list[EvidenceItem]:
    """Sample ``k`` evidence items without replacement using NumPy probabilities."""

    np_generator = np.random.default_rng(generator.randrange(2**32))
    indices = np_generator.choice(
        len(evidence_pool),
        size=k,
        replace=False,
        p=probabilities,
    )
    return [evidence_pool[index] for index in indices]

def _sample_initial_evidence_profile(
    generator: random.Random,
    evidence_pool: Sequence[EvidenceItem],
    k: int,
    n: int,
    probabilities: np.ndarray,
) -> list[list[EvidenceItem]]:
    """Sample the initial private evidence profile for ``n`` agents."""

    return [
        _sample_private_evidence(generator, evidence_pool, k, probabilities)
        for _ in range(n)
    ]

def _coverage_diversity(
    evidence_profile: Sequence[Sequence[EvidenceItem]],
    alternative: Opinion,
    *,
    l: int,
    m: int,
) -> float:
    """Return the fraction of one side's evidence covered by an initial profile."""

    if alternative not in {"a", "b"}:
        raise ValueError("alternative must be 'a' or 'b'.")

    denominator = l if alternative == "a" else m - l
    if denominator == 0:
        raise ValueError(f"Cannot compute coverage diversity for {alternative} with no such evidence.")

    covered_evidence = {
        item
        for evidence in evidence_profile
        for item in evidence
        if item.startswith(alternative)
    }
    return len(covered_evidence) / denominator

def _pairwise_diversity(
    evidence_profile: Sequence[Sequence[EvidenceItem]],
    alternative: Opinion,
    *,
    k: int,
) -> float:
    """Return normalized average pairwise symmetric difference for one side."""

    if alternative not in {"a", "b"}:
        raise ValueError("alternative must be 'a' or 'b'.")
    if len(evidence_profile) < 2:
        raise ValueError("pairwise diversity requires at least two agents.")

    evidence_sets = [
        {item for item in evidence if item.startswith(alternative)}
        for evidence in evidence_profile
    ]
    total_difference = 0
    pair_count = 0

    for first_index, first_evidence in enumerate(evidence_sets[:-1]):
        for second_evidence in evidence_sets[first_index + 1:]:
            total_difference += len(first_evidence ^ second_evidence)
            pair_count += 1

    return total_difference / pair_count / (2 * k)

def _validate_parameters(k: int, l: int, m: int, n: int) -> None:
    """Validate the parameter ranges for one deliberation instance."""

    _validate_evidence_parameters(l=l, m=m)
    if m < 1:
        raise ValueError("m must be at least 1.")
    if n < 1:
        raise ValueError("n must be at least 1.")
    if not 1 <= k <= m:
        raise ValueError("k must satisfy 1 <= k <= m.")

def _validate_evidence_parameters(l: int, m: int) -> None:
    """Validate the global evidence counts."""

    if m < 1:
        raise ValueError("m must be at least 1.")
    if not 0 <= l <= m:
        raise ValueError("l must satisfy 0 <= l <= m.")

def plot_avg_fraction_correct_agents_at_termination_over_n_per_k(
    k_values: Sequence[int],
    l: int,
    m: int,
    n_values: Sequence[int],
    batch_size: int,
    *,
    s: float | None = None,
    band: str = "ci95",
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot one terminal-accuracy curve per ``k`` across a range of ``n`` values.

    The ``band`` parameter controls the shaded interval:
    ``"std"``, ``"sem"``, ``"ci95"``, or ``"iqr"``.
    """

    if not k_values:
        raise ValueError("k_values must not be empty.")
    if not n_values:
        raise ValueError("n_values must not be empty.")
    if band not in {"std", "sem", "ci95", "iqr"}:
        raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'iqr'.")

    generator = _resolve_rng(rng=rng, seed=seed)
    k_list = list(k_values)
    x_values = list(n_values)

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    ax.axhline(0.5, color="gray", linewidth=1.5, linestyle="-", alpha=0.9)
    
    for index, k in enumerate(k_list):
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for n in x_values:
            # For each (k, n) pair, estimate the terminal accuracy distribution over the batch.
            histories = Batch_Deliberation(
                k=k,
                l=l,
                m=m,
                n=n,
                batch_size=batch_size,
                s=s,
                rng=generator,
            )
            terminal_fractions = [_fraction_correct_at_termination(history) for history in histories]
            mean_fraction = mean(terminal_fractions)
            lower, upper = _band_interval(terminal_fractions, band=band)
            means.append(mean_fraction)
            lowers.append(lower)
            uppers.append(upper)

        color = _plot_color(index, palette)
        line = ax.plot(x_values, means, marker="o", linewidth=2, color=color, label=f"$k={k}$")[0]
        ax.fill_between(
            x_values,
            lowers,
            uppers,
            alpha=0.1,
            color=color,
            # label=_band_label(band) if index == 0 else None,
        )

    ax.set_xlabel("$n$")
    ax.set_ylabel("Correct agents at termination")
    ax.set_title(f"Fraction of correct agents at termination ($\\ell={l}$, $m={m}$)")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x_values)
    ax.set_yticks([i/10 for i in range(11)])
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    _save_plot(
        ax.figure,
        "avg-frac-c-ag-vs-n-per-k",
        l=l,
        m=m,
        batch_size=batch_size,
        band=band,
        k_values=k_list,
        n_values=x_values,
    )

    return ax

def plot_avg_fraction_correct_agents_over_rounds_per_k(
    k_values: Sequence[int],
    l: int,
    m: int,
    n: int,
    nr_rounds: int,
    batch_size: int,
    *,
    s: float | None = None,
    band: str = "ci95",
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot one round-by-round accuracy curve per ``k`` over ``nr_rounds`` rounds.

    If a history terminates before a requested round, the final recorded round is reused.
    The ``band`` parameter controls the shaded interval:
    ``"std"``, ``"sem"``, ``"ci95"``, or ``"iqr"``.
    """

    if not k_values:
        raise ValueError("k_values must not be empty.")
    if nr_rounds < 1:
        raise ValueError("nr_rounds must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if band not in {"std", "sem", "ci95", "iqr"}:
        raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'iqr'.")

    generator = _resolve_rng(rng=rng, seed=seed)
    k_list = list(k_values)
    round_values = list(range(nr_rounds))

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    for index, k in enumerate(k_list):
        histories = Batch_Deliberation(
            k=k,
            l=l,
            m=m,
            n=n,
            batch_size=batch_size,
            s=s,
            rng=generator,
        )

        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        ax.axhline(0.5, color="gray", linewidth=1.5, linestyle="-", alpha=0.9)
        
        for round_index in round_values:
            fractions_at_round = [
                _fraction_correct_at_round(history, round_index)
                for history in histories
            ]
            mean_fraction = mean(fractions_at_round)
            lower, upper = _band_interval(fractions_at_round, band=band)
            means.append(mean_fraction)
            lowers.append(lower)
            uppers.append(upper)

        color = _plot_color(index, palette)
        line = ax.plot(round_values, means, marker="o", linewidth=2, color=color, label=f"$k={k}$")[0]
        ax.fill_between(
            round_values,
            lowers,
            uppers,
            alpha=0.1,
            color=color,
            # label=_band_label(band) if index == 0 else None,
        )

    ax.set_xlabel(f"$t$")
    ax.set_ylabel("Fraction of correct agents")
    ax.set_title(f"Fraction of correct agents over rounds ($\\ell={l}$, $m={m}$, $n={n}$)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(round_values)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    _save_plot(
        ax.figure,
        "avg-frac-c-ag-vs-r-per-k",
        l=l,
        m=m,
        batch_size=batch_size,
        band=band,
        k_values=k_list,
        n=n,
        nr_rounds=nr_rounds,
    )

    return ax

def plot_avg_fraction_correct_agents_over_rounds_per_n(
    k: int,
    l: int,
    m: int,
    n_values: Sequence[int],
    nr_rounds: int,
    batch_size: int,
    *,
    s: float | None = None,
    band: str = "ci95",
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot one round-by-round accuracy curve per ``n`` over ``nr_rounds`` rounds.

    If a history terminates before a requested round, the final recorded round is reused.
    The ``band`` parameter controls the shaded interval:
    ``"std"``, ``"sem"``, ``"ci95"``, or ``"iqr"``.
    """

    if not n_values:
        raise ValueError("n_values must not be empty.")
    if nr_rounds < 1:
        raise ValueError("nr_rounds must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if band not in {"std", "sem", "ci95", "iqr"}:
        raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'iqr'.")

    generator = _resolve_rng(rng=rng, seed=seed)
    n_list = list(n_values)
    round_values = list(range(nr_rounds))

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    for index, n in enumerate(n_list):
        histories = Batch_Deliberation(
            k=k,
            l=l,
            m=m,
            n=n,
            batch_size=batch_size,
            s=s,
            rng=generator,
        )

        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for round_index in round_values:
            fractions_at_round = [
                _fraction_correct_at_round(history, round_index)
                for history in histories
            ]
            mean_fraction = mean(fractions_at_round)
            lower, upper = _band_interval(fractions_at_round, band=band)
            means.append(mean_fraction)
            lowers.append(lower)
            uppers.append(upper)

        color = _plot_color(index, palette)
        line = ax.plot(round_values, means, marker="o", linewidth=2, color=color, label=f"$n={n}$")[0]
        ax.fill_between(
            round_values,
            lowers,
            uppers,
            alpha=0.1,
            color=color,
            # label=_band_label(band) if index == 0 else None,
        )

    ax.set_xlabel(f"$t$")
    ax.set_ylabel("Fraction of correct agents")
    ax.set_title(f"Fraction of correct agents over rounds ($\\ell={l}$, $m={m}$, $k={k}$)")
    ax.axhline(0.5, color="gray", linewidth=1.5, linestyle="-", alpha=0.9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(round_values)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    _save_plot(
        ax.figure,
        "avg-frac-c-ag-vs-r-per-n",
        l=l,
        m=m,
        batch_size=batch_size,
        band=band,
        k=k,
        n_values=n_list,
        nr_rounds=nr_rounds,
    )

    return ax

def plot_fraction_correct_decisions_at_termination_over_n_per_k(
    k_values: Sequence[int],
    l: int,
    m: int,
    n_values: Sequence[int],
    batch_size: int,
    *,
    s: float | None = None,
    band: str = "ci95",
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot the terminal fraction of correct collective decisions across ``n`` values.

    A trial counts as correct when the final collective opinion is ``'a'``. The y-value
    at each point is the batch mean of these binary outcomes, one curve per ``k``.
    The ``band`` parameter controls the shaded interval: ``"std"``, ``"sem"``,
    ``"ci95"``, or ``"iqr"``.
    """

    if not k_values:
        raise ValueError("k_values must not be empty.")
    if not n_values:
        raise ValueError("n_values must not be empty.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if band not in {"std", "sem", "ci95", "iqr"}:
        raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'iqr'.")

    generator = _resolve_rng(rng=rng, seed=seed)
    k_list = list(k_values)
    x_values = list(n_values)

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    for index, k in enumerate(k_list):
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for n in x_values:
            histories = Batch_Deliberation(
                k=k,
                l=l,
                m=m,
                n=n,
                batch_size=batch_size,
                s=s,
                rng=generator,
            )
            correct_decisions = [_correct_collective_decision_at_termination(history) for history in histories]
            mean_fraction = mean(correct_decisions)
            lower, upper = _band_interval(correct_decisions, band=band)
            means.append(mean_fraction)
            lowers.append(lower)
            uppers.append(upper)

        color = _plot_color(index, palette)
        line = ax.plot(x_values, means, marker="o", linewidth=2, color=color, label=f"$k={k}$")[0]
        ax.fill_between(
            x_values,
            lowers,
            uppers,
            alpha=0.1,
            color=color,
            # label=_band_label(band) if index == 0 else None,
        )

    ax.set_xlabel("$n$")
    ax.set_ylabel("Correct collective decisions at termination")
    ax.set_title(f"Correct collective decisions at termination ($\\ell={l}$, $m={m}$)")
    ax.axhline(0.5, color="gray", linewidth=1.5, linestyle="-", alpha=0.9)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x_values)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    _save_plot(
        ax.figure,
        "frac-c-colldec-vs-n-per-k",
        l=l,
        m=m,
        batch_size=batch_size,
        band=band,
        k_values=k_list,
        n_values=x_values,
    )

    return ax

def plot_fraction_correct_decisions_over_rounds_per_k(
    k_values: Sequence[int],
    l: int,
    m: int,
    n: int,
    nr_rounds: int,
    batch_size: int,
    *,
    s: float | None = None,
    band: str = "ci95",
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot one round-by-round collective-decision curve per ``k`` over ``nr_rounds`` rounds."""

    if not k_values:
        raise ValueError("k_values must not be empty.")
    if nr_rounds < 1:
        raise ValueError("nr_rounds must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if band not in {"std", "sem", "ci95", "iqr"}:
        raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'iqr'.")

    generator = _resolve_rng(rng=rng, seed=seed)
    k_list = list(k_values)
    round_values = list(range(nr_rounds))

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    for index, k in enumerate(k_list):
        histories = Batch_Deliberation(
            k=k,
            l=l,
            m=m,
            n=n,
            batch_size=batch_size,
            s=s,
            rng=generator,
        )

        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for round_index in round_values:
            correct_decisions_at_round = [
                _correct_collective_decision_at_round(history, round_index)
                for history in histories
            ]
            mean_fraction = mean(correct_decisions_at_round)
            lower, upper = _band_interval(correct_decisions_at_round, band=band)
            means.append(mean_fraction)
            lowers.append(lower)
            uppers.append(upper)

        color = _plot_color(index, palette)
        line = ax.plot(round_values, means, marker="o", linewidth=2, color=color, label=f"$k={k}$")[0]
        ax.fill_between(
            round_values,
            lowers,
            uppers,
            alpha=0.1,
            color=color,
            # label=_band_label(band) if index == 0 else None,
        )

    ax.set_xlabel("$t$")
    ax.set_ylabel("Fraction of correct collective decisions")
    ax.set_title(f"Fraction collective decisions over rounds ($\\ell={l}$, $m={m}$, $n={n}$)")
    ax.axhline(0.5, color="gray", linewidth=1.5, linestyle="-", alpha=0.9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(round_values)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    _save_plot(
        ax.figure,
        "frac-c-colldec-vs-r-per-k",
        l=l,
        m=m,
        batch_size=batch_size,
        band=band,
        k_values=k_list,
        n=n,
        nr_rounds=nr_rounds,
    )

    return ax

def plot_fraction_correct_decisions_over_rounds_per_n(
    k: int,
    l: int,
    m: int,
    n_values: Sequence[int],
    nr_rounds: int,
    batch_size: int,
    *,
    s: float | None = None,
    band: str = "ci95",
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot one round-by-round collective-decision curve per ``n`` over ``nr_rounds`` rounds."""

    if not n_values:
        raise ValueError("n_values must not be empty.")
    if nr_rounds < 1:
        raise ValueError("nr_rounds must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if band not in {"std", "sem", "ci95", "iqr"}:
        raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'iqr'.")

    generator = _resolve_rng(rng=rng, seed=seed)
    n_list = list(n_values)
    round_values = list(range(nr_rounds))

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    for index, n in enumerate(n_list):
        histories = Batch_Deliberation(
            k=k,
            l=l,
            m=m,
            n=n,
            batch_size=batch_size,
            s=s,
            rng=generator,
        )

        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for round_index in round_values:
            correct_decisions_at_round = [
                _correct_collective_decision_at_round(history, round_index)
                for history in histories
            ]
            mean_fraction = mean(correct_decisions_at_round)
            lower, upper = _band_interval(correct_decisions_at_round, band=band)
            means.append(mean_fraction)
            lowers.append(lower)
            uppers.append(upper)

        color = _plot_color(index, palette)
        line = ax.plot(round_values, means, marker="o", linewidth=2, color=color, label=f"$n={n}$")[0]
        ax.fill_between(
            round_values,
            lowers,
            uppers,
            alpha=0.1,
            color=color,
            # label=_band_label(band) if index == 0 else None,
        )

    ax.set_xlabel("$t$")
    ax.set_ylabel("Fraction of correct collective decisions")
    ax.set_title(f"Correct collective decisions over rounds ($\\ell={l}$, $m={m}$, $k={k}$)")
    ax.axhline(0.5, color="gray", linewidth=1.5, linestyle="-", alpha=0.9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(round_values)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    _save_plot(
        ax.figure,
        "frac-c-colldec-vs-r-per-n",
        l=l,
        m=m,
        batch_size=batch_size,
        band=band,
        k=k,
        n_values=n_list,
        nr_rounds=nr_rounds,
    )

    return ax

def plot_fraction_correct_decisions_over_n_per_round(
    k: int,
    l: int,
    m: int,
    n_values: Sequence[int],
    rounds_of_interest: Sequence[int],
    batch_size: int,
    *,
    s: float | None = None,
    band: str = "ci95",
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot correct collective decisions against ``n``, with one curve per round of interest."""

    if not n_values:
        raise ValueError("n_values must not be empty.")
    if not rounds_of_interest:
        raise ValueError("rounds_of_interest must not be empty.")
    if any(round_index < 0 for round_index in rounds_of_interest):
        raise ValueError("rounds_of_interest must contain only non-negative integers.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if band not in {"std", "sem", "ci95", "iqr"}:
        raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'iqr'.")

    generator = _resolve_rng(rng=rng, seed=seed)
    x_values = list(n_values)
    round_list = list(rounds_of_interest)

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    # ax.axhline(1, color="gray", linewidth=2, linestyle="-", alpha=1)

    # For each n, generate exactly one batch and reuse it across all requested rounds.
    histories_by_n = {
        n: Batch_Deliberation(
            k=k,
            l=l,
            m=m,
            n=n,
            batch_size=batch_size,
            s=s,
            rng=generator,
        )
        for n in x_values
    }

    for index, round_index in enumerate(round_list):
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for n in x_values:
            histories = histories_by_n[n]
            correct_decisions = [
                _correct_collective_decision_at_round(history, round_index)
                for history in histories
            ]
            mean_fraction = mean(correct_decisions)
            lower, upper = _band_interval(correct_decisions, band=band)
            means.append(mean_fraction)
            lowers.append(lower)
            uppers.append(upper)

        color = _plot_color(index, palette)
        line = ax.plot(x_values, means, marker="o", linewidth=2, color=color, label=f"$t={round_index}$")[0]
        ax.fill_between(
            x_values,
            lowers,
            uppers,
            alpha=0.1,
            color=color,
        )

    ax.set_xlabel("$n$")
    ax.set_ylabel("Fraction of correct collective decisions")
    ax.set_title(f"$\\ell={l}$, $m={m}$, $k={k}$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(x_values)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend(loc="right")
    _save_plot(
        ax.figure,
        "frac-c-colldec-vs-n-per-r",
        l=l,
        m=m,
        batch_size=batch_size,
        band=band,
        k=k,
        n_values=x_values,
        round_values=round_list,
    )

    return ax

def plot_fraction_correct_decisions_over_k_per_round(
    l: int,
    m: int,
    n: int,
    k_values: Sequence[int],
    rounds_of_interest: Sequence[int],
    batch_size: int,
    *,
    s: float | None = None,
    band: str = "ci95",
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot correct collective decisions against ``k``, with one curve per round of interest."""

    if not k_values:
        raise ValueError("k_values must not be empty.")
    if not rounds_of_interest:
        raise ValueError("rounds_of_interest must not be empty.")
    if any(round_index < 0 for round_index in rounds_of_interest):
        raise ValueError("rounds_of_interest must contain only non-negative integers.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if band not in {"std", "sem", "ci95", "iqr"}:
        raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'iqr'.")

    generator = _resolve_rng(rng=rng, seed=seed)
    x_values = list(k_values)
    round_list = list(rounds_of_interest)

    for k in x_values:
        _validate_parameters(k=k, l=l, m=m, n=n)

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    # For each k, generate exactly one batch and reuse it across all requested rounds.
    histories_by_k = {
        k: Batch_Deliberation(
            k=k,
            l=l,
            m=m,
            n=n,
            batch_size=batch_size,
            s=s,
            rng=generator,
        )
        for k in x_values
    }

    # threshold = m-l
    # ax.axvline(threshold, color="gray", linewidth=2, linestyle="-", alpha=1)
    # ax.text(
    #     threshold-0.4,
    #     0.45,
    #     rf"$m-\ell$",
    #     rotation=90,
    #     color="gray",
    #     va="top",
    #     ha="right",
    #     fontsize=10,
    #     alpha=1,
    #     )
    
    # threshold = m*(m-l-1)/l
    # ax.axvline(threshold, color="gray", linewidth=2, linestyle="-", alpha=1)
    # ax.text(
    #     threshold-0.7,
    #     0.03,
    #     rf"$\frac{{m(m-\ell-1)}}{{\ell}}$",
    #     rotation=90,
    #     color="gray",
    #     va="bottom",
    #     ha="right",
    #     fontsize=12,
    #     alpha=1,
    #     )
    
    # threshold = m*(m-l+1)/l
    # ax.axvline(threshold, color="gray", linewidth=2, linestyle="-", alpha=1)
    # ax.text(
    #     threshold+0.7,
    #     0.03,
    #     rf"$\frac{{m(m-\ell+1)}}{{\ell}}$",
    #     rotation=90,
    #     color="gray",
    #     va="bottom",
    #     ha="left",
    #     fontsize=12,
    #     alpha=1,
    #     )

    for index, round_index in enumerate(round_list):
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for k in x_values:
            histories = histories_by_k[k]
            correct_decisions = [
                _correct_collective_decision_at_round(history, round_index)
                for history in histories
            ]
            mean_fraction = mean(correct_decisions)
            lower, upper = _band_interval(correct_decisions, band=band)
            means.append(mean_fraction)
            lowers.append(lower)
            uppers.append(upper)

        color = _plot_color(index, palette)
        ax.plot(x_values, means, marker="o", linewidth=2, color=color, label=f"$t={round_index}$")
        ax.fill_between(
            x_values,
            lowers,
            uppers,
            alpha=0.1,
            color=color,
        )

    ax.set_xlabel("$k$")
    ax.set_ylabel("Fraction of correct collective decisions")
    ax.set_title(f"$\\ell={l}$, $m={m}$, $n={n}$") # fraction of correct collective decisions
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(x_values[::2])
    ax.set_yticks([i / 10 for i in range(11)])
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend(loc="right")
    _save_plot(
        ax.figure,
        "frac-c-colldec-vs-k-per-r",
        l=l,
        m=m,
        batch_size=batch_size,
        band=band,
        k_values=x_values,
        n=n,
        round_values=round_list,
    )

    return ax

def plot_avg_number_of_rounds_over_n_per_k(
    k_values: Sequence[int],
    l: int,
    m: int,
    n_values: Sequence[int],
    batch_size: int,
    *,
    s: float | None = None,
    band: str = "ci95",
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot the average last disclosure round across ``n`` values, with one curve per ``k``.

    The y-value is the last round index at which some disclosure occurs. Because the
    final recorded round always has no disclosure, this is the second-to-last recorded
    round. For example, if the recorded rounds are ``[0, 1, 2]``, the plotted value is
    ``1``. The ``band`` parameter controls the shaded interval: ``"std"``, ``"sem"``,
    ``"ci95"``, or ``"iqr"``.
    """

    if not k_values:
        raise ValueError("k_values must not be empty.")
    if not n_values:
        raise ValueError("n_values must not be empty.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if band not in {"std", "sem", "ci95", "iqr"}:
        raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'iqr'.")

    generator = _resolve_rng(rng=rng, seed=seed)
    k_list = list(k_values)
    x_values = list(n_values)

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    for index, k in enumerate(k_list):
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for n in x_values:
            histories = Batch_Deliberation(
                k=k,
                l=l,
                m=m,
                n=n,
                batch_size=batch_size,
                s=s,
                rng=generator,
            )
            disclosure_rounds = [_last_disclosure_round_index(history) for history in histories]
            mean_disclosure_round = mean(disclosure_rounds)
            lower, upper = _band_interval(disclosure_rounds, band=band, bounds=(0.0, None))
            means.append(mean_disclosure_round)
            lowers.append(lower)
            uppers.append(upper)

        color = _plot_color(index, palette)
        line = ax.plot(x_values, means, marker="o", linewidth=2, color=color, label=f"$k={k}$")[0]
        ax.fill_between(
            x_values,
            lowers,
            uppers,
            alpha=0.1,
            color=color,
            # label=_band_label(band) if index == 0 else None,
        )

    ax.axhline(2, color="gray", linewidth=2, linestyle="-", alpha=0.5)
    ax.axhline(1, color="gray", linewidth=2, linestyle="-", alpha=0.5)
    ax.set_xlabel("$n$")
    ax.set_ylabel("Average last disclosure round")
    ax.set_title(f"Average last disclosure round ($\\ell={l}$, $m={m}$)")
    ax.set_xticks(x_values)
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    _save_plot(
        ax.figure,
        "avg-nr-r-vs-n-per-k",
        l=l,
        m=m,
        batch_size=batch_size,
        band=band,
        k_values=k_list,
        n_values=x_values,
    )

    return ax

def plot_avg_number_of_rounds_over_k_per_n(
    k_values: Sequence[int],
    l: int,
    m: int,
    n_values: Sequence[int],
    batch_size: int,
    *,
    s: float | None = None,
    band: str = "ci95",
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot the average last disclosure round across ``k`` values, with one curve per ``n``.

    The y-value is the last round index at which some disclosure occurs. Because the
    final recorded round always has no disclosure, this is the second-to-last recorded
    round. The ``band`` parameter controls the shaded interval: ``"std"``, ``"sem"``,
    ``"ci95"``, or ``"iqr"``.
    """

    if not k_values:
        raise ValueError("k_values must not be empty.")
    if not n_values:
        raise ValueError("n_values must not be empty.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if band not in {"std", "sem", "ci95", "iqr"}:
        raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'iqr'.")

    generator = _resolve_rng(rng=rng, seed=seed)
    x_values = list(k_values)
    n_list = list(n_values)

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    for index, n in enumerate(n_list):
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for k in x_values:
            histories = Batch_Deliberation(
                k=k,
                l=l,
                m=m,
                n=n,
                batch_size=batch_size,
                s=s,
                rng=generator,
            )
            disclosure_rounds = [_last_disclosure_round_index(history) for history in histories]
            mean_disclosure_round = mean(disclosure_rounds)
            lower, upper = _band_interval(disclosure_rounds, band=band, bounds=(0.0, None))
            means.append(mean_disclosure_round)
            lowers.append(lower)
            uppers.append(upper)

        color = _plot_color(index, palette)
        line = ax.plot(x_values, means, marker="o", linewidth=2, color=color, label=f"$n={n}$")[0]
        ax.fill_between(
            x_values,
            lowers,
            uppers,
            alpha=0.1,
            color=color,
            # label=_band_label(band) if index == 0 else None,
        )

    ax.axhline(2, color="gray", linewidth=2.5, linestyle="-", alpha=0.5)
    ax.axhline(1, color="gray", linewidth=2.5, linestyle="-", alpha=0.5)
    ax.set_xlabel("$k$")
    ax.set_ylabel("Average last disclosure round")
    ax.set_title(f"Average last disclosure round ($\\ell={l}$, $m={m}$)")
    ax.set_xticks(x_values)
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    _save_plot(
        ax.figure,
        "avg-nr-r-vs-k-per-n",
        l=l,
        m=m,
        batch_size=batch_size,
        band=band,
        k_values=x_values,
        n_values=n_list,
    )

    return ax

def plot_bound(l, m, k_values):
    """Plot the theoretical threshold for correct collective decisions at termination."""

    if not k_values:
        raise ValueError("k_values must not be empty.")

    x_values = list(k_values)
    y_values = [k * (l/m) + sqrt(k * (m - k + 1)* log(2)/(2 * m)) for k in x_values]

    _, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x_values, y_values, marker="o", linewidth=2, color="black", label = "Bound")
    ax.plot(x_values, x_values, label = "$k$")
    ax.hlines(m-l, x_values[0], x_values[-1], color="gray", linewidth=2, linestyle="-", alpha=0.5, label="$m-\\ell$")
    ax.vlines(0.5 * (m - sqrt(log(2)) * 0.5), 0, x_values[-1], color="gray", linewidth=2, linestyle="-", alpha=0.5)

    ax.set_xlabel("$k$")
    ax.set_ylabel("Bound")
    ax.set_title(f"Bound ($\\ell={l}$, $m={m}$)")
    ax.set_xticks(x_values)
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    plt.show()
    # return ax

def plot_avg_number_of_successes_over_s_per_k(
    k_values: Sequence[int],
    l: int,
    m: int,
    *,
    nr_s_values: int,
    nr_draws: int,
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot average number of a-evidence items sampled across salience values."""

    if not k_values:
        raise ValueError("k_values must not be empty.")
    if nr_s_values < 2:
        raise ValueError("nr_s_values must be at least 2.")
    if nr_draws < 1:
        raise ValueError("nr_draws must be at least 1.")

    generator = _resolve_rng(rng=rng, seed=seed)
    evidence_pool = generate_evidence(m=m, l=l)
    s_values = np.linspace(0.0, 1.0, nr_s_values)
    print(f"Salience values: {s_values}")
    k_list = list(k_values)

    for k in k_list:
        _validate_parameters(k=k, l=l, m=m, n=1)

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    baseline_s = l / m
    ax.axvline(baseline_s, color="gray", linewidth=1.5, linestyle=":", alpha=0.9)
    ax.text(
        baseline_s - 0.01,
        1,
        rf"$\frac{{\ell}}{{m}}$",
        rotation=90,
        color="gray",
        va="top",
        ha="right",
        fontsize=14,
    )

    for index, k in enumerate(k_list):
        color = _plot_color(index, palette)
        baseline = k * l / m
        ax.axhline(k * (l / m), color=color, linewidth=1.5, linestyle="-", alpha=0.75)
        ax.text(
            1,
            baseline + 1.5,
            rf"$k \cdot \frac{{\ell}}{{m}}$",
            rotation=0,
            color=color,
            va="top",
            ha="right",
            fontsize=12,
        )

    for index, k in enumerate(k_list):
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for s in s_values:
            salience = _resolve_salience(s=float(s), l=l, m=m)
            _validate_salience_for_sampling(salience, k=k, l=l, m=m)
            probabilities = _evidence_saliences(evidence_pool, s=salience)
            success_counts = [
                sum(
                    1
                    for item in _sample_private_evidence(
                        generator,
                        evidence_pool,
                        k,
                        probabilities,
                    )
                    if item.startswith("a")
                )
                for _ in range(nr_draws)
            ]
            means.append(mean(success_counts))
            lower, upper = _band_interval(success_counts, band="ci95", bounds=(0.0, float(k)))
            lowers.append(lower)
            uppers.append(upper)

        color = _plot_color(index, palette)
        ax.fill_between(
            s_values,
            lowers,
            uppers,
            alpha=0.1,
            color=color,
        )
        ax.plot(s_values, means, marker="o", linewidth=2, color=color, label=f"$k={k}$")
        
    ax.set_xlabel("$s$")
    ax.set_ylabel(f"Average initial draw for $a$")
    ax.set_title(f"$\\ell={l}$, $m={m}$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    _save_plot(
        ax.figure,
        "avg-nr-successes-vs-s-per-k",
        l=l,
        m=m,
        batch_size=nr_draws,
        band="none",
        k_values=k_list,
    )

    return ax

def plot_fraction_correct_decisions_at_termination_over_s(
    k: int,
    l: int,
    m: int,
    n: int,
    batch_size: int,
    *,
    nr_s_values: int,
    band: str = "ci95",
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot terminal collective accuracy across salience values."""

    _validate_parameters(k=k, l=l, m=m, n=n)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if nr_s_values < 2:
        raise ValueError("nr_s_values must be at least 2.")
    if band not in {"std", "sem", "ci95", "iqr"}:
        raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'iqr'.")

    generator = _resolve_rng(rng=rng, seed=seed)
    s_values = np.linspace(0.0, 1.0, nr_s_values)

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    baseline_s = l / m
    # ax.axhline(0.5, color="gray", linewidth=1.5, linestyle="-", alpha=0.45)
    ax.axvline(baseline_s, color="gray", linewidth=1.5, linestyle=":", alpha=0.9)
    ax.text(
        baseline_s - 0.01,
        0.96,
        rf"$\frac{{\ell}}{{m}}$",
        rotation=90,
        color="gray",
        va="top",
        ha="right",
        fontsize=12,
    )

    means: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []

    for s in s_values:
        salience = _resolve_salience(s=float(s), l=l, m=m)
        _validate_salience_for_sampling(salience, k=k, l=l, m=m)
        histories = Batch_Deliberation(
            k=k,
            l=l,
            m=m,
            n=n,
            batch_size=batch_size,
            s=salience,
            rng=generator,
        )
        correct_decisions = [
            _correct_collective_decision_at_termination(history)
            for history in histories
        ]
        means.append(mean(correct_decisions))
        lower, upper = _band_interval(correct_decisions, band=band)
        lowers.append(lower)
        uppers.append(upper)

    color = _plot_color(0, palette)
    ax.fill_between(
        s_values,
        lowers,
        uppers,
        alpha=0.1,
        color=color,
    )
    ax.plot(s_values, means, marker="o", linewidth=2, color=color, label=f"$k={k}$")
    ax.set_xlabel("$s$")
    ax.set_ylabel("Correct collective decisions at termination")
    ax.set_title(f"$\\ell={l}$, $m={m}$, $n={n}$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks([i / 10 for i in range(11)])
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    _save_plot(
        ax.figure,
        "frac-c-colldec-term-vs-s",
        l=l,
        m=m,
        batch_size=batch_size,
        band=band,
        k=k,
        n=n,
    )

    return ax

def plot_avg_number_of_rounds_over_s(
    k: int,
    l: int,
    m: int,
    n: int,
    batch_size: int,
    *,
    nr_s_values: int,
    band: str = "ci95",
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot average last disclosure round across salience values."""

    _validate_parameters(k=k, l=l, m=m, n=n)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if nr_s_values < 2:
        raise ValueError("nr_s_values must be at least 2.")
    if band not in {"std", "sem", "ci95", "iqr"}:
        raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'iqr'.")

    generator = _resolve_rng(rng=rng, seed=seed)
    s_values = np.linspace(0.0, 1.0, nr_s_values)

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    baseline_s = l / m
    ax.axvline(baseline_s, color="gray", linewidth=1.5, linestyle=":", alpha=0.9)
    ax.text(
        baseline_s - 0.01,
        0.1,
        rf"$\frac{{\ell}}{{m}}$",
        rotation=90,
        color="gray",
        va="top",
        ha="right",
        fontsize=12,
    )

    means: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []

    for s in s_values:
        salience = _resolve_salience(s=float(s), l=l, m=m)
        _validate_salience_for_sampling(salience, k=k, l=l, m=m)
        histories = Batch_Deliberation(
            k=k,
            l=l,
            m=m,
            n=n,
            batch_size=batch_size,
            s=salience,
            rng=generator,
        )
        rounds = [_last_disclosure_round_index(history) for history in histories]
        means.append(mean(rounds))
        lower, upper = _band_interval(rounds, band=band, bounds=(0.0, None))
        lowers.append(lower)
        uppers.append(upper)

    color = _plot_color(0, palette)
    ax.fill_between(
        s_values,
        lowers,
        uppers,
        alpha=0.1,
        color=color,
    )
    ax.plot(s_values, means, marker="o", linewidth=2, color=color, label=f"$k={k}$")
    ax.set_xlabel("$s$")
    ax.set_ylabel("Average round at termination")
    ax.set_title(f"$\\ell={l}$, $m={m}$, $n={n}$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1, 1.25])
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    _save_plot(
        ax.figure,
        "avg-nr-rounds-vs-s",
        l=l,
        m=m,
        batch_size=batch_size,
        band=band,
        k=k,
        n=n,
    )

    return ax

def plot_avg_coverage_diversity_over_s(
    k: int,
    l: int,
    m: int,
    n: int,
    batch_size: int,
    *,
    nr_s_values: int,
    band: str = "ci95",
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot average initial coverage diversity for a- and b-evidence over salience."""

    _validate_parameters(k=k, l=l, m=m, n=n)
    if not 0 < l < m:
        raise ValueError("coverage diversity for both alternatives requires 0 < l < m.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if nr_s_values < 2:
        raise ValueError("nr_s_values must be at least 2.")
    if band not in {"std", "sem", "ci95", "iqr"}:
        raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'iqr'.")

    generator = _resolve_rng(rng=rng, seed=seed)
    evidence_pool = generate_evidence(m=m, l=l)
    s_values = np.linspace(0.0, 1.0, nr_s_values)

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    baseline_s = l / m
    ax.axvline(baseline_s, color="gray", linewidth=1.5, linestyle=":", alpha=0.9)
    ax.text(
        baseline_s - 0.01,
        0.96,
        rf"$\frac{{\ell}}{{m}}$",
        rotation=90,
        color="gray",
        va="top",
        ha="right",
        fontsize=12,
    )

    for index, alternative in enumerate(("a", "b")):
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for s in s_values:
            salience = _resolve_salience(s=float(s), l=l, m=m)
            _validate_salience_for_sampling(salience, k=k, l=l, m=m)
            probabilities = _evidence_saliences(evidence_pool, s=salience)
            diversity_values = [
                _coverage_diversity(
                    _sample_initial_evidence_profile(
                        generator,
                        evidence_pool,
                        k,
                        n,
                        probabilities,
                    ),
                    alternative,
                    l=l,
                    m=m,
                )
                for _ in range(batch_size)
            ]
            means.append(mean(diversity_values))
            lower, upper = _band_interval(diversity_values, band=band)
            lowers.append(lower)
            uppers.append(upper)

        color = _plot_color(index, palette)
        ax.fill_between(
            s_values,
            lowers,
            uppers,
            alpha=0.1,
            color=color,
        )
        ax.plot(
            s_values,
            means,
            marker="o",
            linewidth=2,
            color=ALTERNATIVE_COLORS[alternative],
            label=rf"$div_{alternative}$",
        )

    ax.set_xlabel("$s$")
    ax.set_ylabel("Average coverage diversity")
    ax.set_title(f"$\\ell={l}$, $m={m}$, $n={n}$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks([i / 10 for i in range(11)])
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    _save_plot(
        ax.figure,
        "avg-coverage-diversity-vs-s",
        l=l,
        m=m,
        batch_size=batch_size,
        band=band,
        k=k,
        n=n,
    )

    return ax

def plot_avg_pairwise_diversity_over_s(
    k: int,
    l: int,
    m: int,
    n: int,
    batch_size: int,
    *,
    nr_s_values: int,
    band: str = "ci95",
    palette: Sequence[str] | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
    ax=None,
):
    """Plot average pairwise diversity for a- and b-evidence over salience."""

    _validate_parameters(k=k, l=l, m=m, n=n)
    if n < 2:
        raise ValueError("pairwise diversity requires at least two agents.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if nr_s_values < 2:
        raise ValueError("nr_s_values must be at least 2.")
    if band not in {"std", "sem", "ci95", "iqr"}:
        raise ValueError("band must be one of: 'std', 'sem', 'ci95', 'iqr'.")

    generator = _resolve_rng(rng=rng, seed=seed)
    evidence_pool = generate_evidence(m=m, l=l)
    s_values = np.linspace(0.0, 1.0, nr_s_values)

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    baseline_s = l / m
    ax.axvline(baseline_s, color="gray", linewidth=1.5, linestyle=":", alpha=0.9)
    ax.text(
        baseline_s - 0.01,
        0.96,
        rf"$\frac{{\ell}}{{m}}$",
        rotation=90,
        color="gray",
        va="top",
        ha="right",
        fontsize=12,
    )

    for index, alternative in enumerate(("a", "b")):
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for s in s_values:
            salience = _resolve_salience(s=float(s), l=l, m=m)
            _validate_salience_for_sampling(salience, k=k, l=l, m=m)
            probabilities = _evidence_saliences(evidence_pool, s=salience)
            diversity_values = [
                _pairwise_diversity(
                    _sample_initial_evidence_profile(
                        generator,
                        evidence_pool,
                        k,
                        n,
                        probabilities,
                    ),
                    alternative,
                    k=k,
                )
                for _ in range(batch_size)
            ]
            means.append(mean(diversity_values))
            lower, upper = _band_interval(diversity_values, band=band)
            lowers.append(lower)
            uppers.append(upper)

        color = _plot_color(index, palette)
        ax.fill_between(
            s_values,
            lowers,
            uppers,
            alpha=0.1,
            color=color,
        )
        ax.plot(
            s_values,
            means,
            marker="o",
            linewidth=2,
            color=color,
            label=rf"$D_{alternative}$",
        )

    ax.set_xlabel("$s$")
    ax.set_ylabel("Average pairwise diversity")
    ax.set_title(f"$\\ell={l}$, $m={m}$, $n={n}$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks([i / 10 for i in range(11)])
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    _save_plot(
        ax.figure,
        "avg-pairwise-diversity-vs-s",
        l=l,
        m=m,
        batch_size=batch_size,
        band=band,
        k=k,
        n=n,
    )

    return ax

def _band_interval(
    values: Sequence[float],
    band: str,
    bounds: tuple[float | None, float | None] = (0.0, 1.0),
) -> tuple[float, float]:
    """Return the plotting interval for one batch statistic."""

    center = mean(values)
    spread = pstdev(values)

    if band == "std":
        lower = center - spread
        upper = center + spread
    elif band == "sem":
        margin = spread / sqrt(len(values))
        lower = center - margin
        upper = center + margin
    elif band == "ci95":
        margin = 1.96 * spread / sqrt(len(values))
        lower = center - margin
        upper = center + margin
    else:
        sorted_values = sorted(values)
        lower = _percentile(sorted_values, 0.25)
        upper = _percentile(sorted_values, 0.75)

    lower_bound, upper_bound = bounds
    if lower_bound is not None:
        lower = max(lower_bound, lower)
    if upper_bound is not None:
        upper = min(upper_bound, upper)
    return lower, upper


def _plot_color(index: int, palette: Sequence[str] | None) -> str:
    """Return the color to use for a plotted series."""

    active_palette = palette if palette else DEFAULT_PLOT_PALETTE
    if not active_palette:
        raise ValueError("palette must not be empty.")
    return active_palette[index % len(active_palette)]


def _save_plot(
    figure,
    plot_name: str,
    *,
    l: int,
    m: int,
    batch_size: int,
    band: str,
    k_values: Sequence[int] | None = None,
    n_values: Sequence[int] | None = None,
    round_values: Sequence[int] | None = None,
    k: int | None = None,
    n: int | None = None,
    nr_rounds: int | None = None,
) -> Path:
    """Save a figure under a readable, parameterized filename."""

    DEFAULT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    parts = [
        plot_name,
        f"m-{m}",
        f"l-{l}",
        # f"batch-{batch_size}",
    ]
    # if k_values is not None:
    #     parts.append(f"k-{_value_summary(k_values)}")
    # if n_values is not None:
    #     parts.append(f"n-{_value_summary(n_values)}")
    # if round_values is not None:
    #     parts.append(f"t-{_value_summary(round_values)}")
    # if k is not None:
    #     parts.append(f"k-{k}")
    # if n is not None:
    #     parts.append(f"n-{n}")
    # if nr_rounds is not None:
    #     parts.append(f"rounds-{nr_rounds}")

    filename = "-".join(parts) + ".png"
    output_path = DEFAULT_PLOT_DIR / filename
    figure.savefig(output_path, dpi=DPI, bbox_inches="tight", transparent=True)
    return output_path

def _value_summary(values: Sequence[int]) -> str:
    """Summarize a sequence compactly for filenames."""

    if not values:
        return "empty"
    if len(values) <= 5:
        return "-".join(str(value) for value in values)

    steps = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    if steps and all(step == steps[0] for step in steps):
        return f"{values[0]}-to-{values[-1]}-step-{steps[0]}"
    return f"{values[0]}-to-{values[-1]}-len-{len(values)}"

def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Compute a simple linear-interpolated percentile from sorted data."""

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * q
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + weight * (upper - lower)

def _band_label(band: str) -> str:
    """Return a human-readable legend label for the chosen uncertainty band."""

    labels = {
        "std": "±1 standard deviation",
        "sem": "±1 standard error",
        "ci95": "95% confidence interval",
        "iqr": "interquartile range",
    }
    return labels[band]

def _fraction_correct_at_round(history: Sequence[RoundState], round_index: int) -> float:
    """Compute the fraction of agents supporting 'a' at ``round_index``.

    If the history ends earlier, reuse the last recorded round.
    """

    effective_round = history[min(round_index, len(history) - 1)]
    correct_agents = sum(1 for agent in effective_round.agents if agent.opinion == "a")
    return correct_agents / len(effective_round.agents)

def _correct_collective_decision_at_round(history: Sequence[RoundState], round_index: int) -> float:
    """Return 1.0 iff the collective opinion at ``round_index`` is 'a'; else 0.0.

    If the history ends earlier, reuse the last recorded round.
    """

    effective_round = history[min(round_index, len(history) - 1)]
    return 1.0 if effective_round.collective_opinion == "a" else 0.0

def _last_disclosure_round_index(history: Sequence[RoundState]) -> int:
    """Return the last round index at which some disclosure occurs."""

    if len(history) == 1:
        return 0
    return history[-2].round_index

# from deliberation_simulation import (
#     Deliberation,
#     Batch_Deliberation,
#     plot_avg_fraction_correct_agents_at_termination,
# )

# history = Deliberation(k=3, l=4, m=7, n=5, seed=7)
# batch = Batch_Deliberation(k=3, l=4, m=7, n=5, batch_size=100, seed=7)

if __name__ == "__main__":
    # history = Deliberation(k=3, l=4, m=7, n=5, seed=None)
    # print("Deliberation history:")
    # for round_state in history:
    #     print(f"Round {round_state.round_index}:")
    #     print(f"  Disclosure occurred: {round_state.disclosure}")
    #     print(f"  Disclosed this round: {round_state.disclosed_this_round}")
    #     print(f"  Public evidence: {round_state.public_evidence}")
    #     print(f"  Collective opinion: {round_state.collective_opinion}")
    #     for agent_index, agent in enumerate(round_state.agents):
    #         print(f"    Agent {agent_index + 1}: {agent.evidence} --> {agent.opinion}")
    # batch = Batch_Deliberation(k=3, l=4, m=7, n=5, batch_size=100, seed=7)

    batch_size = 1000
    m = 100 
    l = 51
    k = 21
    k_values = [11, 13, 15]
    n = 100
    nr_rounds = 5

    # plot_avg_fraction_correct_agents_at_termination_over_n_per_k(
    #     k_values=k_values,
    #     l=l,
    #     m=m,
    #     n_values=range(1, n+1, 5),
    #     batch_size=batch_size,
    #     band="ci95",
    #     seed=7,
    # )

    # plot_avg_fraction_correct_agents_over_rounds_per_k(
    #     k_values=k_values,
    #     l=l,
    #     m=m,
    #     n=n,
    #     nr_rounds=nr_rounds,
    #     batch_size=batch_size,
    #     band="ci95",
    #     seed=7,
    # )

    # plot_avg_fraction_correct_agents_over_rounds_per_n(
    #     k=k,
    #     l=l,
    #     m=m,
    #     n_values=[5, 15, 51, 91],
    #     nr_rounds=nr_rounds,
    #     batch_size=batch_size,
    #     band="ci95",
    #     seed=7,
    # )

    # plot_fraction_correct_decisions_at_termination_over_n_per_k(
    #     k_values=k_values,
    #     l=l,
    #     m=m,
    #     n_values=range(1, n, 5),
    #     batch_size=batch_size,
    #     band="ci95",
    #     seed=7,
    # )

    # plot_fraction_correct_decisions_over_rounds_per_k(
    #     k_values=k_values,
    #     l=l,
    #     m=m,
    #     n=n,
    #     nr_rounds=nr_rounds,
    #     batch_size=batch_size,
    #     band="ci95",
    #     seed=7,
    # )

    # plot_fraction_correct_decisions_over_rounds_per_n(
    #     k=k,
    #     l=l,
    #     m=m,
    #     n_values=[5, 25, 55],
    #     nr_rounds=nr_rounds,
    #     batch_size=batch_size,
    #     band="ci95",
    #     seed=7,
    # )

    k, l, m, n, batch_size = 17, 52, 100, 111, 1000
    plot_fraction_correct_decisions_over_n_per_round(
        k=k,
        l=l,
        m=m,
        n_values=range(1, n, 8),
        rounds_of_interest=[0],
        batch_size=batch_size,
        band="ci95",
        seed=7,
    )

    l, m, n, batch_size = 51, 100, 71, 100
    # plot_fraction_correct_decisions_over_k_per_round(
    #     l=l,
    #     m=m,
    #     n=n,
    #     k_values=range(1, m+1, 4),
    #     rounds_of_interest=[0],
    #     batch_size=batch_size,
    #     seed=7,
    # )

    l, m, n, batch_size = 52, 100, 71, 200
    # plot_avg_number_of_rounds_over_n_per_k(
    #     k_values=[17],
    #     l=l,
    #     m=m,
    #     n_values=range(1, 101, 5),
    #     batch_size=batch_size,
    #     band="ci95",
    #     seed=7,
    # )

    # l, m, n = 65, 100, 101
    # plot_avg_number_of_rounds_over_k_per_n(
    #     k_values=[5, 25, 45, 65, 85],
    #     l=l,
    #     m=m,
    #     n_values=[51, 101, 1001],
    #     batch_size=batch_size,
    #     band="ci95",
    #     seed=7,
    # ) 

    ### salience plots
    l, m, n, k, batch_size, nr_s_values = 60, 100, 71, 15, 1000, 31
    # plot_avg_number_of_successes_over_s_per_k(
    #     k_values=[k],
    #     l=l,
    #     m=m,
    #     nr_s_values=nr_s_values,
    #     nr_draws=batch_size,
    #     seed=7,
    # )

    # plot_fraction_correct_decisions_at_termination_over_s(
    #     k=k,
    #     l=l,
    #     m=m,
    #     n=n,
    #     batch_size=batch_size,
    #     nr_s_values=nr_s_values,
    #     seed=7,
    # )

    # plot_avg_number_of_rounds_over_s(
    #     k=k,
    #     l=l,
    #     m=m,
    #     n=n,
    #     batch_size=batch_size,
    #     nr_s_values=nr_s_values,
    #     seed=7,
    # )

    # plot_avg_coverage_diversity_over_s(
    #     k,
    #     l,
    #     m,
    #     n,
    #     batch_size=batch_size,
    #     nr_s_values=nr_s_values,
    #     seed=7,
    # )

    # plot_avg_pairwise_diversity_over_s(
    #     k=k,
    #     l=l,
    #     m=m,
    #     n=n,
    #     batch_size=batch_size,
    #     nr_s_values=nr_s_values,
    #     seed=7,
    # )