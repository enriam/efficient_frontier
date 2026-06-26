# Class Asset: financial asset

from dataclasses import dataclass


@dataclass(frozen=False)
class Asset:
    name: str
    avg: float  # mean
    std: float  # standard deviation
