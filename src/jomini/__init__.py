"""Jomini: Lanchester-type combat models for simulating battles."""

from jomini import cdb90
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
    "cdb90",
]

__version__ = "1.0.0"
