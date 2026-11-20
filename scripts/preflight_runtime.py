"""Fail fast before tests that intentionally exercise the live local stack."""
import json

from nusalens.api.app import readiness


def main() -> int:
    state = readiness()
    print(json.dumps(state, indent=2))
    unavailable = {name: value for name, value in state.items() if value != "ok"}
    if unavailable:
        print("Live integration tests require `make bootstrap` to complete successfully.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
