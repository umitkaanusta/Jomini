"""Jomini: Lanchester-type combat models for simulating battles."""

from jomini.battle import Battle
from jomini.models import GeneralLaw, LanchesterModel, LinearLaw, LogarithmicLaw, SquareLaw
from jomini.result import BattleResult, Outcome

__all__ = [
    "Battle",
    "BattleResult",
    "GeneralLaw",
    "LanchesterModel",
    "LinearLaw",
    "LogarithmicLaw",
    "Outcome",
    "SquareLaw",
]

__version__ = "1.0.0"
