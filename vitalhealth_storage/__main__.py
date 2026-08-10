"""Initialise the VitalHealth shared database schema.

Run after DATABASE_URL has been configured:
    python -m vitalhealth_storage
"""

from . import get_store


def main() -> None:
    store = get_store()
    store.initialize()
    print("VitalHealth shared database schema is ready.")


if __name__ == "__main__":
    main()
