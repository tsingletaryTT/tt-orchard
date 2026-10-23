# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Model files shared by the card-use tests: a chat template and a README cut from real shapes."""
TEMPLATE = """\
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort') }}
    {%- endif %}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\\n' }}
{%- endif %}
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\\n\\n</think>\\n\\n' }}
{%- else %}
    {{- '<think>\\n' }}
{%- endif %}
"""

README_WITH_USE = """\
---
license: cc-by-nc-4.0
base_model:
- Qwen/Qwen3.8-27B
---
# Hemmingway-1

A long-horizon agent model.

## Intended use
AREX-like models are intended for long-horizon agents and tool-augmented research.

## License
CC BY-NC 4.0.
"""
