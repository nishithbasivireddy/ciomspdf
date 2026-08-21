from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

from src.field_schema import (
    ALL_FIELDS,
    CHECKBOX_SCHEMA,
    FIELD_SCHEMA,
)


_COMPONENT_ROOT = Path(__file__).parent
_FRONTEND_ROOT = _COMPONENT_ROOT / "frontend"
_LAYOUT_PATH = _FRONTEND_ROOT / "layout.json"

_cioms_component = components.declare_component(
    "cioms_expected_form",
    path=str(_FRONTEND_ROOT),
)


def blank_expected_values() -> dict[str, str]:
    values = {
        field: ""
        for field in FIELD_SCHEMA
    }

    values.update(
        {
            field: "Off"
            for field in CHECKBOX_SCHEMA
        }
    )

    return values


def normalize_expected_values(
    values: dict[str, Any] | None,
) -> dict[str, str]:
    normalized = blank_expected_values()

    if not values:
        return normalized

    for field in ALL_FIELDS:
        if field not in values:
            continue

        raw_value = values[field]

        if field in CHECKBOX_SCHEMA:
            normalized[field] = (
                "Yes"
                if str(raw_value).strip().casefold()
                == "yes"
                else "Off"
            )
        else:
            normalized[field] = (
                ""
                if raw_value is None
                else str(raw_value)
            )

    return normalized


def validate_expected_values(
    values: dict[str, Any],
) -> dict[str, list[str]]:
    supplied_keys = set(values)
    expected_keys = set(ALL_FIELDS)

    missing_keys = sorted(
        expected_keys - supplied_keys
    )

    unexpected_keys = sorted(
        supplied_keys - expected_keys
    )

    invalid_checkboxes = sorted(
        field
        for field in CHECKBOX_SCHEMA
        if field in values
        and str(values[field]).strip() not in {
            "Yes",
            "Off",
        }
    )

    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "invalid_checkboxes": invalid_checkboxes,
    }


def load_layout() -> dict[str, Any]:
    if not _LAYOUT_PATH.is_file():
        raise FileNotFoundError(
            f"CIOMS layout is missing: "
            f"{_LAYOUT_PATH}"
        )

    layout = json.loads(
        _LAYOUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    if layout.get("widget_count") != 38:
        raise ValueError(
            "CIOMS layout must contain exactly "
            "38 widgets."
        )

    return layout


def render_cioms_expected_form(
    initial_values: dict[str, Any] | None = None,
    key: str = "cioms_expected_form",
) -> dict[str, str]:
    layout = load_layout()
    values = normalize_expected_values(
        initial_values
    )

    result = _cioms_component(
        layout=layout,
        field_schema=FIELD_SCHEMA,
        checkbox_schema=CHECKBOX_SCHEMA,
        values=values,
        key=key,
        default=values,
    )

    if not isinstance(result, dict):
        return values

    return normalize_expected_values(result)
