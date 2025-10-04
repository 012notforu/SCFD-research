import pytest


def pytest_collection_modifyitems(config, items):
    markexpr = getattr(config.option, "markexpr", "") or ""
    if "slow" in markexpr:
        return
    skip_slow = pytest.mark.skip(reason="use -m slow to run slow tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
