"""Shared PostgreSQL persistence primitives for the VitalHealth apps."""

from .store import ClinicalStore, get_store

__all__ = ["ClinicalStore", "get_store"]
