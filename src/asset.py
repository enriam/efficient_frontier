# Class Asset: financial asset

from dataclasses import dataclass


@dataclass(frozen=False)
class Asset:
    name: str
    ret: float  # mean
    volat: float  # stdev
