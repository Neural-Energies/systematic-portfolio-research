from pathlib import Path

import pytest

from systematic_research.institutional_eda import diagnostic_inventory, load_development_bars


def test_diagnostic_inventory_has_exactly_one_hundred_checks() -> None:
    inventory = diagnostic_inventory()

    assert len(inventory) == 100
    assert inventory.index.min() == 1
    assert inventory.index.max() == 100
    assert inventory["family"].nunique() == 10


def test_development_loader_rejects_missing_development_partition(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Development minute-return partitions"):
        load_development_bars(tmp_path)
