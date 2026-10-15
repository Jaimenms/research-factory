"""Casos em JSON: `case_from_json`, `load_json`, `validate_case`, `case_to_json`; herança em `extends`."""

from .spec import SCHEMA_ID, Case, case_from_json, case_to_json, load_json, validate_case

__all__ = ["SCHEMA_ID", "Case", "case_from_json", "case_to_json", "load_json", "validate_case"]
