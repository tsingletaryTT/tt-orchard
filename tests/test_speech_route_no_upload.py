# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The speech route never packages, copies or uploads a model's weights (operator decision, 2026-11-23: it serves and checks what a bundle's author published).

Two kinds of check. A static one reads the route's code and fails on an import, a name or a command that belongs to packaging or uploading. A behavioural one runs the
runner's own refusal over the commands the route writes. The static check has to be able to fail, so each rule is run against a source that breaks it."""
import ast
import shlex
import textwrap
from pathlib import Path

import pytest

from orchard import runner, speech_route as sr

REPO = Path(__file__).resolve().parent.parent
ROUTE_MODULES = ("speech_route.py", "speech_hw.py", "speech_reference_run.py", "speech_models.py", "speech_check.py")
FORBIDDEN_NAMES = {"upload", "upload_file", "upload_folder", "upload_large_folder", "create_commit", "create_repo", "push_to_hub", "HfApi", "hf_hub_upload", "publish",
                   "can", "can_audio", "package", "package_cmd", "full_port_bundle", "text_package", "audio_package", "qwen35_package", "granite_package"}
FORBIDDEN_MODULES = {"huggingface_hub", "orchard.can", "orchard.can_audio", "orchard.package", "orchard.package_cmd", "orchard.audio_package", "orchard.text_package",
                     "orchard.qwen35_package", "orchard.granite_package", "orchard.acestep_package", "orchard.full_port_bundle"}
FORBIDDEN_PROGRAMS = {"hf", "huggingface-cli", "git", "docker", "tt-orchard", "scp", "rsync", "curl", "wget"}
FORBIDDEN_VERBS = {"push", "publish", "can", "upload", "pull"}
ALLOWED_TT_MODEL_VERBS = {"serve", "stop"}


def problems(tree: ast.AST, label: str) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            out.append(f"{label}: the name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            out.append(f"{label}: the attribute {node.attr}")
        elif isinstance(node, ast.Import):
            out += [f"{label}: imports {a.name}" for a in node.names if a.name in FORBIDDEN_MODULES or a.name.split(".")[0] == "huggingface_hub"]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod in FORBIDDEN_MODULES or mod.split(".")[0] == "huggingface_hub":
                out.append(f"{label}: imports from {mod}")
            out += [f"{label}: imports {mod}.{a.name}" for a in node.names if f"{mod}.{a.name}" in FORBIDDEN_MODULES or (mod == "orchard" and a.name in FORBIDDEN_NAMES)]
        elif isinstance(node, (ast.List, ast.Tuple)) and node.elts:
            head = node.elts[0]
            if isinstance(head, ast.Constant) and head.value in FORBIDDEN_PROGRAMS:
                out.append(f"{label}: a command list starts with {head.value!r}")
            if len(node.elts) > 1 and isinstance(node.elts[1], ast.Constant) and node.elts[1].value in FORBIDDEN_VERBS:
                out.append(f"{label}: a command list has the verb {node.elts[1].value!r}")
            if isinstance(head, ast.Name) and head.id == "tt_model" and len(node.elts) > 1:
                verb = node.elts[1].value if isinstance(node.elts[1], ast.Constant) else None
                if verb not in ALLOWED_TT_MODEL_VERBS:
                    out.append(f"{label}: a tt-model command with the verb {verb!r}")
    return out


def source_of(path: str) -> ast.AST:
    return ast.parse((REPO / "orchard" / path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("module", ROUTE_MODULES)
def test_a_route_module_has_no_packaging_upload_or_publish_in_it(module):
    assert problems(source_of(module), module) == []


def functions_named(path: Path, prefix: str, tree=None) -> ast.AST:
    tree = tree or ast.parse(path.read_text(encoding="utf-8"))
    picked = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith(prefix)]
    assert picked, f"no function named {prefix}* in {path.name}"
    return ast.Module(body=picked, type_ignores=[])


def test_the_supervisors_speech_methods_have_no_packaging_upload_or_publish_in_them():
    assert problems(functions_named(REPO / "orchard" / "supervisor.py", "_speech"), "supervisor._speech*") == []


def test_the_bundle_builders_speech_functions_have_no_packaging_upload_or_publish_in_them():
    assert problems(functions_named(REPO / "orchard" / "skills" / "operator-bundle-templates" / "build_bundle.py", "speech_"), "build_bundle.speech_*") == []


def test_the_stage_table_of_the_route_has_no_stage_that_packages():
    from orchard.stages import SPEECH_TABLE
    assert SPEECH_TABLE[7].skip and "never packages or republishes" in SPEECH_TABLE[7].skip
    assert all(spec.draft is None for spec in SPEECH_TABLE.values())


def test_the_only_tt_model_verbs_in_the_route_are_serve_and_stop():
    seen = set()
    for node in ast.walk(source_of("speech_hw.py")):
        if isinstance(node, (ast.List, ast.Tuple)) and node.elts and isinstance(node.elts[0], ast.Name) and node.elts[0].id == "tt_model":
            seen.add(node.elts[1].value)
    assert seen == ALLOWED_TT_MODEL_VERBS


# ---- the checks can fail ---------------------------------------------------------------------------------------------------

BREAKS = {
    "an upload call": "def f(api):\n    api.upload_folder('x')\n",
    "a publish name": "def f():\n    publish()\n",
    "importing the packager": "from orchard import can\n",
    "importing a module of the packager": "import orchard.package_cmd\n",
    "importing huggingface_hub": "from huggingface_hub import HfApi\n",
    "an hf command list": "cmd = ['hf', 'upload', 'x']\n",
    "a tt-orchard command list": "cmd = ['tt-orchard', 'can', 'x']\n",
    "a push verb": "cmd = [prog, 'push', 'x']\n",
    "a tt-model push": "cmd = [tt_model, 'push', bundle]\n",
    "a tt-model publish": "cmd = [tt_model, 'publish', bundle]\n",
}


@pytest.mark.parametrize("name", sorted(BREAKS))
def test_each_rule_of_the_static_check_catches_a_source_that_breaks_it(name):
    assert problems(ast.parse(BREAKS[name]), "break"), name


def test_a_source_that_does_what_the_route_does_passes_the_static_check():
    good = "cmd = [tt_model, 'serve', bundle]\nstop = [tt_model, 'stop', bundle]\nrun = [python, '-I', script, '--mode', 'asr']\n"
    assert problems(ast.parse(good), "good") == []


# ---- the runner refuses what the route would never write -------------------------------------------------------------------


def test_the_runner_accepts_the_command_the_route_writes(tmp_path):
    runner.check_string(sr.hw_command(tmp_path / "s2"), tmp_path)


@pytest.mark.parametrize("command", ["tt-model publish apande-TT/x", "tt-model push apande-TT/x", "git push origin main", "hf upload someone/x ."])
def test_the_runner_refuses_commands_that_publish(tmp_path, command):
    with pytest.raises(runner.Denied):
        runner.check_string(command, tmp_path)
