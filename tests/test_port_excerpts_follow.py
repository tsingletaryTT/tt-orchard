# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Reference excerpts follow the classes the kept classes instantiate (run9 of the tiny Qwen4 rehearsal).

The first excerpt file named the classes the config and the usual block names select. The tiny Qwen4 model's decoder layer also builds a
gated residual, a PLE layer and an n-gram embedding, whose names match no rule, so they were not in the file and the outside helper wrote
"hyper-connection mixer and PLE: source not in excerpts". `port_excerpts` now adds a same-file class that a kept class calls by name
(depth 2), after the classes the rules chose, inside the same caps, and the file says why each was added."""
from orchard import port_excerpts
from test_port_facts import CONFIG

SOURCE = '''
import torch
from torch import nn


class ToyExpRMSNorm(nn.Module):
    def forward(self, x):
        return x


class ToyExpGatedResidual(nn.Module):
    """Mixes the stream of a layer with its input."""
    def __init__(self, config):
        super().__init__()
        self.norm = ToyExpInnerGate(config)

    def forward(self, stream, update):
        return stream + self.norm(update)


class ToyExpInnerGate(nn.Module):
    def forward(self, x):
        return x


class ToyExpPLELayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.table = ToyExpNGramEmbedding(config)

    def forward(self, hidden_states, ids):
        return hidden_states + self.table(ids)


class ToyExpNGramEmbedding(nn.Module):
    def forward(self, ids):
        return ids


class ToyExpVisionPatchEmbed(nn.Module):
    def forward(self, x):
        return x


class ToyExpNeverUsed(nn.Module):
    def forward(self, x):
        return x


class ToyExpDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.mixer = ToyExpGatedResidual(config)
        self.ple = ToyExpPLELayer(config, layer_idx)
        self.patch = ToyExpVisionPatchEmbed()
        self.fn = nn.Linear(2, 2)

    def forward(self, hidden_states, **kwargs):
        return self.mixer(hidden_states, self.ple(hidden_states, kwargs["ids"]))


class ToyExpForCausalLM(nn.Module):
    def forward(self, input_ids, **kwargs):
        return input_ids
'''


def test_a_class_a_kept_class_instantiates_is_added_with_the_reason():
    r = port_excerpts.build(CONFIG, SOURCE)
    assert "ToyExpGatedResidual" in r.classes and "ToyExpPLELayer" in r.classes
    assert "instantiated by ToyExpDecoderLayer" in r.text
    assert "self.norm(update)" in r.text                       # the added class's forward is in the file


def test_the_follow_goes_two_levels_deep_and_no_further():
    r = port_excerpts.build(CONFIG, SOURCE)
    assert "ToyExpNGramEmbedding" in r.classes                  # DecoderLayer -> PLELayer -> NGramEmbedding
    assert "ToyExpInnerGate" in r.classes                       # DecoderLayer -> GatedResidual -> InnerNorm
    deeper = SOURCE.replace("class ToyExpNGramEmbedding(nn.Module):\n    def forward(self, ids):\n        return ids\n",
                            "class ToyExpNGramEmbedding(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
                            "        self.deep = ToyExpDeep()\n\n    def forward(self, ids):\n        return ids\n\n\n"
                            "class ToyExpDeep(nn.Module):\n    def forward(self, x):\n        return x\n")
    assert "ToyExpDeep" not in port_excerpts.build(CONFIG, deeper).classes


def test_a_vision_class_and_an_unused_class_are_not_followed_into_a_text_file():
    r = port_excerpts.build(CONFIG, SOURCE)
    assert "ToyExpVisionPatchEmbed" not in r.classes and "ToyExpNeverUsed" not in r.classes
    assert "ToyExpVisionPatchEmbed" in port_excerpts.build({**CONFIG, "vision_config": {"depth": 1}}, SOURCE).classes


def test_followed_classes_come_after_the_classes_the_rules_chose():
    r = port_excerpts.build(CONFIG, SOURCE)
    assert r.classes.index("ToyExpDecoderLayer") < r.classes.index("ToyExpGatedResidual")
    assert r.classes.index("ToyExpRMSNorm") < r.classes.index("ToyExpGatedResidual")


def test_followed_classes_leave_first_when_the_class_count_is_capped(monkeypatch):
    monkeypatch.setattr(port_excerpts, "MAX_CLASSES", 3)
    r = port_excerpts.build(CONFIG, SOURCE)
    assert len(r.classes) <= 3 and "ToyExpDecoderLayer" in r.classes
    assert any("ToyExpGatedResidual" in x and "more than 3 matching classes" in x for x in r.left_out)


def test_a_followed_class_that_does_not_fit_the_total_is_listed_in_the_header(monkeypatch):
    found = None
    for cap in range(2500, 14000, 100):
        monkeypatch.setattr(port_excerpts, "MAX_TOTAL_BYTES", cap)
        r = port_excerpts.build(CONFIG, SOURCE)
        if "ToyExpDecoderLayer" in r.classes and any(c in r.classes for c in ("ToyExpGatedResidual", "ToyExpPLELayer")) is False \
                and any("total cap" in x and ("ToyExpGatedResidual" in x or "ToyExpPLELayer" in x) for x in r.left_out):
            found = r
            break
    assert found is not None, "no cap leaves the rule-chosen classes in and the followed ones out"
    header = found.text.split("## ", 1)[0]
    assert "Left out or cut:" in header and ("ToyExpGatedResidual" in header or "ToyExpPLELayer" in header)


def test_a_class_two_kept_classes_both_build_is_added_once():
    twice = SOURCE.replace("        return input_ids\n", "        self.again = ToyExpGatedResidual(None)\n        return input_ids\n")
    twice = twice.replace("class ToyExpForCausalLM(nn.Module):\n    def forward", "class ToyExpForCausalLM(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.again = ToyExpGatedResidual(None)\n\n    def forward")
    r = port_excerpts.build(CONFIG, twice)
    assert r.classes.count("ToyExpGatedResidual") == 1 and r.text.count("## ToyExpGatedResidual") == 1
