"""The strategy registry.

One place that knows every strategy the platform can run, whether it came from
a Python class or a YAML file, so nothing has to be discovered by grepping.

Registration is by ``(id, version)``.  Registering the same pair twice is an
error rather than an overwrite: two different strategies quietly sharing an
identity would make every backtest result ambiguous, which defeats the whole
versioning scheme.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading.core.instrument import InstrumentId
from trading.observability.logging import get_logger
from trading.strategies.base import LifecycleStatus, Strategy, StrategySpec
from trading.strategies.builtin import BuyAndHold, SmaCross
from trading.strategies.config_strategy import load_strategy

__all__ = ["RegisteredStrategy", "StrategyRegistry", "default_registry"]

log = get_logger(__name__)


class DuplicateStrategyError(ValueError):
    pass


class StrategyNotFoundError(KeyError):
    pass


@dataclass(frozen=True, slots=True)
class RegisteredStrategy:
    spec: StrategySpec
    factory: Callable[[], Strategy]
    origin: str
    """Where it came from — a module path or a config file — for the manifest."""

    @property
    def key(self) -> tuple[str, str]:
        return (self.spec.id, self.spec.version)

    def build(self) -> Strategy:
        return self.factory()


class StrategyRegistry:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], RegisteredStrategy] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[RegisteredStrategy]:
        return iter(sorted(self._entries.values(), key=lambda e: e.key))

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def register(
        self, factory: Callable[[], Strategy], *, origin: str = "python"
    ) -> RegisteredStrategy:
        instance = factory()
        entry = RegisteredStrategy(spec=instance.spec, factory=factory, origin=origin)
        if entry.key in self._entries:
            existing = self._entries[entry.key]
            raise DuplicateStrategyError(
                f"{entry.spec.id} version {entry.spec.version} is already registered "
                f"from {existing.origin}. Bump the version rather than shadowing it — "
                f"two strategies sharing an identity make every result ambiguous."
            )
        self._entries[entry.key] = entry
        return entry

    def load_directory(self, directory: Path | str, *, pattern: str = "*.yaml") -> int:
        """Register every config strategy in a directory. Returns how many loaded.

        A malformed file is reported and skipped rather than aborting the load,
        so one bad config does not hide every good one.
        """
        directory = Path(directory)
        if not directory.exists():
            return 0

        loaded = 0
        for path in sorted(directory.glob(pattern)):
            try:
                strategy = load_strategy(path)
                self.register(lambda s=strategy: s, origin=str(path))  # type: ignore[misc]
                loaded += 1
            except Exception as exc:  # one bad config must not hide the rest
                log.warning("strategy_config_invalid", path=str(path), error=str(exc))
        return loaded

    def get(self, strategy_id: str, version: str | None = None) -> RegisteredStrategy:
        """Fetch by id, defaulting to the highest registered version."""
        candidates = [e for e in self._entries.values() if e.spec.id == strategy_id]
        if not candidates:
            known = sorted({e.spec.id for e in self._entries.values()})
            raise StrategyNotFoundError(
                f"No strategy {strategy_id!r}. Registered: {known or 'none'}"
            )
        if version is not None:
            for entry in candidates:
                if entry.spec.version == version:
                    return entry
            versions = sorted(e.spec.version for e in candidates)
            raise StrategyNotFoundError(
                f"{strategy_id} has no version {version!r}. Available: {versions}"
            )
        return max(candidates, key=lambda e: _version_key(e.spec.version))

    def build(self, strategy_id: str, version: str | None = None) -> Strategy:
        return self.get(strategy_id, version).build()

    def by_status(self, status: LifecycleStatus) -> list[RegisteredStrategy]:
        return [e for e in self if e.spec.lifecycle_status is status]

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "id": e.spec.id,
                "version": e.spec.version,
                "name": e.spec.name,
                "status": str(e.spec.lifecycle_status),
                "universe": ", ".join(str(i) for i in e.spec.universe),
                "content_hash": e.spec.content_hash,
                "origin": e.origin,
            }
            for e in self
        ]


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def default_registry(config_dir: Path | str = "configs/strategies") -> StrategyRegistry:
    """A registry with the built-in Python strategies plus any config strategies."""
    registry = StrategyRegistry()
    reliance = InstrumentId("NSE", "RELIANCE")
    registry.register(lambda: BuyAndHold(reliance), origin="trading.strategies.builtin")
    registry.register(lambda: SmaCross(reliance), origin="trading.strategies.builtin")
    registry.load_directory(config_dir)
    return registry
