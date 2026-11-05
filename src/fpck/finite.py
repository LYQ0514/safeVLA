"""Exact finite-state FPCK solver used as theoretical ground truth.

States and actions must be hashable. `successors(s,a)` returns every successor
allowed by the disturbance model. Intervention cost must be 0 or 1. The solver
returns K[h][b], the states winning with at most h steps and b interventions.
"""
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Hashable, Iterable, Mapping, Sequence, Set, Tuple

State = Hashable
Action = Hashable

@dataclass(frozen=True)
class FiniteProductGame:
    states: FrozenSet[State]
    actions: Mapping[State, Tuple[Action, ...]]
    successors: Callable[[State, Action], Iterable[State]]
    intervention_cost: Callable[[State, Action], int]
    accepting: FrozenSet[State]
    rejecting: FrozenSet[State]

    def validate(self) -> None:
        if self.accepting & self.rejecting:
            raise ValueError("accepting and rejecting sets overlap")
        if not (self.accepting | self.rejecting) <= self.states:
            raise ValueError("terminal sets must be subsets of states")
        for s in self.states:
            for a in self.actions.get(s, ()):
                c = self.intervention_cost(s, a)
                if c not in (0, 1):
                    raise ValueError("intervention cost must be binary")
                succ = tuple(self.successors(s, a))
                if not succ or not set(succ) <= self.states:
                    raise ValueError("successors must be nonempty and in states")

def solve_fpck(game: FiniteProductGame, horizon: int, budget: int):
    """Exact robust dynamic program plus witness actions."""
    game.validate()
    if horizon < 0 or budget < 0:
        raise ValueError("horizon and budget must be non-negative")
    K = [[set(game.accepting) for _ in range(budget + 1)] for _ in range(horizon + 1)]
    witness: Dict[Tuple[int, int, State], Action] = {}
    for h in range(1, horizon + 1):
        for b in range(budget + 1):
            win = set(game.accepting)
            for s in game.states - game.rejecting - game.accepting:
                for a in game.actions.get(s, ()):
                    c = game.intervention_cost(s, a)
                    if c <= b and all(ns in K[h - 1][b - c] for ns in game.successors(s, a)):
                        win.add(s); witness[(h, b, s)] = a; break
            K[h][b] = win
    frozen = tuple(tuple(frozenset(x) for x in row) for row in K)
    return frozen, witness

def solve_physical_viability(game: FiniteProductGame) -> FrozenSet[State]:
    """Greatest robust controlled-invariant subset outside rejection."""
    current: Set[State] = set(game.states - game.rejecting)
    while True:
        nxt = {s for s in current if any(
            all(ns in current for ns in game.successors(s, a))
            for a in game.actions.get(s, ()))}
        if nxt == current:
            return frozenset(current)
        current = nxt
