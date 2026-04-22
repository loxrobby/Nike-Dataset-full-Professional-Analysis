from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NikePalette:
    black: str = "#111111"
    white: str = "#FFFFFF"
    gray: str = "#A0A0A0"
    light_gray: str = "#F5F5F5"
    accent: str = "#FF6A00"  # Nike-inspired orange accent
    accent_2: str = "#39FF14"  # neon green accent (optional)


PALETTE = NikePalette()


def format_currency(value: float, currency_symbol: str = "$") -> str:
    if value is None:
        return "-"
    return f"{currency_symbol}{value:,.2f}"


def format_percent(value: float, decimals: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{decimals}f}%"

