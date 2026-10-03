"""Test-run hooks."""
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Name the gozer that ran in the report of a failing contract test.

    The contract tests run whatever gozer checkout is configured, possibly with uncommitted edits
    (see the module docstring of test_gozer_contract.py), so a failure needs to say which one.
    """
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed and item.module.__name__ == "test_gozer_contract":
        report.sections.append(("gozer under test", item.module.gozer_identity()))
