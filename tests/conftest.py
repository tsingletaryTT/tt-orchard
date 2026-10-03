"""Test-run hooks and shared fixtures."""
import pytest


class StubTools:
    """A directory of stub binaries that stand in for tools a test must never really run.

    Each stub appends its name and arguments to `called.log` in the stub directory and exits 1.
    `calls()` returns those lines, so a test can assert that no stub ran.
    """
    NAMES = ("tt-smi", "git", "gozer", "docker", "hf", "huggingface-cli", "tt-model", "tt", "ssh",
             "scp", "rsync", "curl", "wget")

    def __init__(self, root):
        self.dir = root / "stub-bin"
        self.dir.mkdir()
        self.log = self.dir / "called.log"
        for name in self.NAMES:
            stub = self.dir / name
            stub.write_text(f'#!/bin/sh\necho "{name} $*" >> "{self.log}"\nexit 1\n')
            stub.chmod(0o755)

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []


@pytest.fixture
def stub_tools(tmp_path, monkeypatch):
    """PATH becomes the stub directory followed by /usr/bin:/bin. agent_env copies PATH, so every
    shell an agent or a hardware test starts in this test finds the stubs first. The real
    tt-smi, gozer, hf and tt-model live in ~/.local/bin and a venv, which are then off PATH."""
    stubs = StubTools(tmp_path)
    monkeypatch.setenv("PATH", f"{stubs.dir}:/usr/bin:/bin")
    return stubs


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
