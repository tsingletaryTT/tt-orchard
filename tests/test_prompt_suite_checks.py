# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The checks that read one reply of the prompt suite for the failure shapes the tenstorrent `qualitative-check` skill lists:
wrong language, prompt echo, mechanical repetition, doubled subwords, control-token leakage, cross-request leakage, a repeated
or corrupt first token, gibberish, and an empty or failed reply.

The healthy fixtures are replies a real served model gave (the AREX-2 and Qwen3.8-27B speed runs of 2026-10-23, greedy,
thinking on). Nothing may be flagged on them. The unhealthy fixtures are built by hand in the shapes the skill names: no chip
has produced a failing reply of each kind, so they show that a check fires on its shape, not that real failures look like it.
"""
import pytest

from orchard import prompt_suite as ps

# Real replies (first 200 characters of a speed-run reply). They are the model's reasoning text, as the server returned it.
HEALTHY_REASONING = [
    "We need to write a text message to a landlord about a broken boiler. The user wants a concise, polite, and clear message. "
    "We should include key details: what's broken, when it happened, any immediate",
    "We need to fix the retry helper in src/utils/retry.py. The current code:\n\n```python\ndef retry(fn, attempts=5, base=0.5):\n"
    "    delay = base\n    for i in range(attempts):\n        try:\n            return ",
    "We need to create tests/test_retry.py with four pytest tests for the retry helper in src/utils/retry.py. First, we need to "
    "inspect the existing retry helper to understand its API. Let's read src/utils",
]


def entry(content="", reasoning="", **extra):
    return {"id": "chat-capital", "mode": "chat", "content": content, "reasoning": reasoning, "tool_calls": [],
            "finish_reason": "stop", "error": None, **extra}


def prompt(**extra):
    return {"id": "chat-capital", "kind": "chat", "expect_script": "Latin",
            "messages": [{"role": "user", "content": "What is the capital of France, and what is one famous building there? "
                                                      "Answer in two sentences."}], **extra}


def kinds(e, p=None, **kw):
    return sorted(f["kind"] for f in ps.findings(e, p if p is not None else prompt(), **kw))


@pytest.mark.parametrize("text", HEALTHY_REASONING)
def test_a_real_healthy_reply_has_no_finding_as_reasoning(text):
    assert kinds(entry(reasoning=text)) == []


@pytest.mark.parametrize("text", HEALTHY_REASONING)
def test_a_real_healthy_reply_has_no_finding_as_content(text):
    assert kinds(entry(content=text)) == []


def test_a_plain_two_sentence_answer_has_no_finding():
    assert kinds(entry(content="The capital of France is Paris. One famous building there is the Eiffel Tower.")) == []


def test_the_judged_text_is_the_content_and_falls_back_to_the_reasoning():
    assert ps.judged_text(entry(content="answer", reasoning="thinking")) == "answer"
    assert ps.judged_text(entry(content="  \n", reasoning="thinking")) == "thinking"
    assert ps.judged_text(entry()) == ""


# ---- empty or failed ------------------------------------------------------------------------------------------------

def test_a_reply_with_no_text_and_no_tool_call_is_empty():
    assert kinds(entry()) == ["empty_output"]


def test_a_tool_call_alone_is_not_empty():
    assert kinds(entry(tool_calls=[{"name": "get_weather", "arguments": "{}"}])) == []


def test_reasoning_alone_is_not_empty_even_when_it_was_cut_off():
    assert kinds(entry(reasoning="We need to think about this for a long time", finish_reason="length")) == []


def test_a_request_that_failed_is_a_finding_and_nothing_else():
    assert kinds(entry(error="HTTP 400: chat template is not set")) == ["request_failed"]


# ---- mechanical repetition ------------------------------------------------------------------------------------------

def test_one_six_word_phrase_repeated_more_than_three_times_is_a_loop():
    text = "Sure. " + "the quick brown fox jumps over " * 5
    assert kinds(entry(content=text)) == ["mechanical_repetition"]


def test_a_phrase_repeated_three_times_is_not_a_loop():
    assert kinds(entry(content="the quick brown fox jumps over " * 3 + "the lazy dog.")) == []


def test_one_long_line_repeated_on_four_lines_in_a_row_is_a_loop():
    assert kinds(entry(content="Here you go:\n" + "I will now answer the question.\n" * 4)) == ["mechanical_repetition"]


def test_closing_braces_on_four_lines_in_a_row_are_code_and_not_a_loop():
    assert kinds(entry(content="if a {\n if b {\n  if c {\n   if d {\n   }\n   }\n   }\n   }\n")) == []


def test_a_short_repeated_unit_that_fills_a_long_stretch_is_a_loop():
    assert kinds(entry(content="Answer: " + "ab" * 40)) == ["mechanical_repetition"]


def test_a_short_repeated_unit_over_a_short_stretch_is_not():
    assert kinds(entry(content="Answer: " + "ha" * 6)) == []


def test_ordinary_repeated_words_are_not_a_loop():
    assert kinds(entry(content="It is very very good, and the very best of the very few I have seen.")) == []


# ---- doubled subwords -----------------------------------------------------------------------------------------------

def test_three_words_with_a_doubled_syllable_are_a_finding():
    assert kinds(entry(content="The capital is ParisParis and the buildingbuilding there is the EiffelEiffel Tower, a sightsight.")) \
        == ["doubled_subwords"]


def test_one_doubled_word_is_not_a_finding():
    assert kinds(entry(content="The city shows one odd word, sightsight, and nothing else wrong.")) == []


def test_a_legitimate_doubled_word_does_not_count_toward_three():
    assert kinds(entry(content="Mama and papa ate a bonbon, a couscous and a murmur of tartar.")) == []


# ---- prompt echo ----------------------------------------------------------------------------------------------------

def test_a_reply_that_begins_with_the_question_is_an_echo():
    q = prompt()["messages"][-1]["content"]
    assert kinds(entry(content=q + " Paris.")) == ["prompt_echo"]


def test_an_echo_ignores_case_and_spacing():
    q = prompt()["messages"][-1]["content"].upper().replace(" ", "  ")
    assert kinds(entry(content=q)) == ["prompt_echo"]


def test_a_reply_that_quotes_the_question_later_is_not_an_echo():
    q = prompt()["messages"][-1]["content"]
    assert kinds(entry(content="Paris. You asked: " + q)) == []


def test_reasoning_that_restates_the_question_is_not_an_echo():
    q = prompt()["messages"][-1]["content"]
    assert kinds(entry(content="Paris.", reasoning="The user asks: " + q)) == []


def test_a_question_shorter_than_the_echo_length_is_never_an_echo():
    short = prompt(messages=[{"role": "user", "content": "Capital of France?"}])
    assert kinds(entry(content="Capital of France? Paris."), short) == []


# ---- control tokens and tool markup ---------------------------------------------------------------------------------

@pytest.mark.parametrize("leak", ["<|im_end|>", "<|im_start|>assistant", "<|endoftext|>", "<|fim_prefix|>"])
def test_a_special_token_in_the_reply_is_a_leak(leak):
    assert kinds(entry(content=f"Paris is the capital. {leak}")) == ["control_token_leak"]


def test_a_special_token_recorded_by_the_template_is_a_leak_even_if_it_has_no_pipes():
    assert kinds(entry(content="Paris is the capital. [SEP] extra"), specials=["[SEP]"]) == ["control_token_leak"]


def test_a_leak_in_the_reasoning_counts_too():
    assert kinds(entry(content="Paris.", reasoning="hmm <|im_end|> hmm")) == ["control_token_leak"]


def test_think_tags_are_not_a_leak():
    assert kinds(entry(content="Paris.", reasoning="<think>thinking</think>"), specials=["<think>", "</think>"]) == []


def test_a_short_special_token_does_not_match_inside_a_word():
    assert kinds(entry(content="The capital is Paris, near the Seine."), specials=["e", "is"]) == []


def test_tool_call_markup_in_the_content_is_a_leak_because_the_parser_did_not_take_it():
    text = '<tool_call>\n<function=get_weather>\n<parameter=city>Lisbon</parameter>\n</function>\n</tool_call>'
    assert kinds(entry(content=text)) == ["control_token_leak"]


def test_a_parsed_tool_call_with_empty_content_is_clean():
    assert kinds(entry(tool_calls=[{"name": "get_weather", "arguments": '{"city": "Lisbon"}'}])) == []


# ---- leakage between requests ---------------------------------------------------------------------------------------

MARKERS = {"multi-turn": "Zephyrine", "summarize": "Lindfors"}


def test_a_marker_of_another_prompt_in_the_reply_is_a_leak():
    assert kinds(entry(content="The capital is Paris, said Zephyrine."), markers=MARKERS) == ["cross_request_leak"]


def test_a_markers_own_prompt_may_use_it():
    e = entry(content="Your name is Zephyrine and you like green tea.", id="multi-turn")
    assert kinds(e, prompt(id="multi-turn"), markers=MARKERS) == []


def test_a_marker_inside_a_longer_word_is_not_a_leak():
    assert kinds(entry(content="Zephyrines are not a thing."), markers=MARKERS) == []


def test_the_marker_match_is_case_sensitive():
    assert kinds(entry(content="The zephyrine wind blew."), markers=MARKERS) == []


# ---- first token ----------------------------------------------------------------------------------------------------

def test_the_same_word_four_times_at_the_start_is_a_repeated_first_token():
    assert kinds(entry(content="the the the the capital is Paris")) == ["repeated_first_token"]


def test_repeats_with_punctuation_and_case_between_still_count():
    assert kinds(entry(content="Paris, paris. PARIS! Paris is the capital.")) == ["repeated_first_token"]


def test_three_repeats_are_not_enough():
    assert kinds(entry(content="Yes, yes, yes, it is Paris and the Eiffel Tower.")) == []


@pytest.mark.parametrize("start", ["�Paris is the capital.", "́Paris is the capital.", "​Paris is the capital.",
                                   "\x07Paris is the capital.", "!!!Paris is the capital.", ",,,Paris is the capital."])
def test_a_corrupt_first_character_is_a_finding(start):
    assert kinds(entry(content=start)) == ["corrupt_first_token"]


@pytest.mark.parametrize("start", ["```python\nprint(1)\n```", "---\nParis", "# Paris\nThe capital", "**Paris** is the capital.",
                                   "\n\nParis is the capital.", "  Paris is the capital."])
def test_markdown_and_whitespace_at_the_start_are_not_corrupt(start):
    assert kinds(entry(content=start)) == []


# ---- gibberish ------------------------------------------------------------------------------------------------------

def test_text_with_too_few_readable_characters_is_gibberish():
    assert kinds(entry(content="Paris \ufffd\ufffd\u2593\u2592\ufffd\u2591\ufffd\ufffd\u2593\u2592\u2591\ufffd\ufffd\u2593\u2592 is")) \
        == ["gibberish"]


def test_consonant_soup_is_gibberish():
    assert kinds(entry(content="krtp wxzq bdfgh qwrtp zzxkd mnbvc ghjkl rtlps xcvbn lkjhg The capital")) == ["gibberish"]


def test_a_few_words_without_vowels_are_not_gibberish():
    assert kinds(entry(content="Rhythm and crypts and the strengths of the city, in the Eiffel Tower, are well known by now.")) == []


def test_code_with_punctuation_is_not_gibberish():
    code = "def f(x):\n    return {k: [v * 2 for v in vs] for k, vs in x.items() if len(vs) > 0}\n"
    assert kinds(entry(content=code)) == []


# ---- wrong language -------------------------------------------------------------------------------------------------

RU = prompt(id="non-english", expect_script="Cyrillic",
            messages=[{"role": "user", "content": "Какая столица Франции? Ответь одним предложением по-русски."}])


def test_an_english_answer_to_a_russian_question_is_the_wrong_language():
    assert kinds(entry(content="The capital of France is Paris, a city on the Seine."), RU) == ["wrong_language"]


def test_a_russian_answer_is_the_right_language():
    assert kinds(entry(content="Столицей Франции является Париж, город на берегах Сены."), RU) == []


def test_chinese_in_an_answer_that_should_be_latin_is_the_wrong_language():
    assert kinds(entry(content="巴黎是法国的首都，埃菲尔铁塔是那里著名的建筑。")) == ["wrong_language"]


def test_a_name_in_another_script_inside_a_latin_answer_is_fine():
    assert kinds(entry(content="The capital of France is Paris (Париж in Russian), and the Eiffel Tower is there.")) == []


def test_reasoning_alone_is_not_judged_for_language():
    # A model that thinks in English before a Russian answer is normal; with no answer yet there is nothing to judge.
    assert kinds(entry(reasoning="We need to answer in Russian: the capital of France is Paris."), RU) == []


def test_a_reply_with_no_letters_has_no_language():
    assert kinds(entry(content="56 + 12 = 68\n"), prompt()) == []


# ---- more than one at once, and the shape of a finding --------------------------------------------------------------

def test_one_reply_can_have_several_findings_and_each_has_a_kind_and_a_short_detail():
    text = "<|im_end|> " + "the quick brown fox jumps over " * 5
    found = ps.findings(entry(content=text), prompt())
    assert sorted(f["kind"] for f in found) == ["control_token_leak", "mechanical_repetition"]
    assert all(isinstance(f["detail"], str) and 0 < len(f["detail"]) <= 200 for f in found)


def test_every_kind_the_checks_can_return_is_listed_and_the_blocking_ones_are_among_them():
    from orchard.defaults import SUITE_BLOCKING_KINDS
    assert set(SUITE_BLOCKING_KINDS) <= set(ps.KINDS)
    assert set(ps.KINDS) >= {"empty_output", "request_failed", "mechanical_repetition", "doubled_subwords", "prompt_echo",
                             "control_token_leak", "cross_request_leak", "repeated_first_token", "corrupt_first_token",
                             "gibberish", "wrong_language"}


# ---- cases added after the first mutation run -----------------------------------------------------------------------

def test_a_repeated_line_with_blank_lines_between_is_still_a_loop():
    assert kinds(entry(content="I will now answer the question.\n\n" * 4)) == ["mechanical_repetition"]


def test_a_separator_of_punctuation_is_not_a_loop_however_long():
    assert kinds(entry(content="Title\n" + "= " * 40 + "\nBody text of the answer follows here.")) == []


def test_reasoning_that_begins_with_the_question_is_not_an_echo_when_there_is_no_answer_yet():
    q = prompt()["messages"][-1]["content"]
    assert kinds(entry(reasoning=q + " We should say Paris.")) == []


def test_tool_markup_in_the_reasoning_is_not_a_leak():
    # A model may discuss the format while thinking. Only the answer text is a leak, because that is what a client receives.
    assert kinds(entry(content="Paris is the capital.", reasoning="The format is <tool_call> then <function=name>.")) == []


def test_a_two_word_reply_that_repeats_a_word_is_not_a_repeated_first_token():
    assert kinds(entry(content="Paris. Paris.")) == []


def test_a_very_short_reply_is_not_judged_on_the_share_of_readable_characters():
    assert kinds(entry(content="☃☃☃")) == []


LONG_Y_WORDS = "rhythm crypts myths lynch nymph gypsy sylph tryst"


def test_words_whose_only_vowel_is_y_are_words():
    assert kinds(entry(content=f"The words are {LONG_Y_WORDS} and nothing else of note here.")) == []


def test_a_few_vowelless_words_among_many_ordinary_long_words_are_not_gibberish():
    bare = "bdfgh qwrtp zzxkd mnbvc ghjkl rtlps xcvbn lkjhg"
    ordinary = ("capital building question sentence however between another example reasoning country neighbour mountain "
                "language morning evening picture station library kitchen weather holiday teacher student doctor market "
                "garden bridge forest island valley desert planet engine window letter number season animal flower stone").split()
    assert len(ordinary) >= 40
    assert kinds(entry(content=bare + " " + " ".join(ordinary))) == []


def test_seven_vowelless_words_are_too_few_to_call_gibberish():
    assert kinds(entry(content="bdfgh qwrtp zzxkd mnbvc ghjkl rtlps xcvbn")) == []


def test_a_very_short_answer_in_the_wrong_script_is_too_short_to_judge():
    assert kinds(entry(content="Oui"), RU) == []


@pytest.mark.parametrize("script,text", [
    ("Greek", "Η πρωτεύουσα της Γαλλίας είναι το Παρίσι."),
    ("Arabic", "عاصمة فرنسا هي باريس وهي مدينة جميلة."),
    ("Hebrew", "בירת צרפת היא פריז והיא עיר יפה מאוד."),
    ("Hangul", "프랑스의 수도는 파리이며 아름다운 도시입니다."),
    ("Thai", "เมืองหลวงของฝรั่งเศสคือปารีสซึ่งเป็นเมืองที่สวยงาม"),
    ("Devanagari", "फ्रांस की राजधानी पेरिस है और यह एक सुंदर शहर है।"),
    ("CJK", "巴黎是法国的首都，埃菲尔铁塔是那里著名的建筑。"),
    ("CJK", "パリはフランスの首都で、エッフェル塔があります。"),
    ("CJK", "パリハフランスノシュトデス。"),
])
def test_each_writing_system_is_recognised_as_itself_and_as_wrong_for_latin(script, text):
    p = prompt(expect_script=script)
    assert kinds(entry(content=text), p) == []
    assert kinds(entry(content=text)) == ["wrong_language"]


def test_a_short_line_repeated_four_times_with_blank_lines_between_is_a_loop():
    # Four repeats of a four-word line: the 6-word phrase rule does not fire (a phrase appears three times), so only the
    # line rule can see it, and blank lines between the repeats must not hide it.
    assert kinds(entry(content="Wait, let me check.\n\n" * 4)) == ["mechanical_repetition"]


def test_a_short_unit_repeated_eight_times_over_a_short_stretch_is_not_a_loop():
    assert kinds(entry(content="Answer: " + "ab" * 10)) == []


def test_a_two_character_special_token_is_not_searched_for():
    assert kinds(entry(content="def f():\n    return {}\n"), specials=["{}"]) == []


def test_a_special_token_that_is_a_plain_word_is_not_searched_for():
    assert kinds(entry(content="The capital is Paris, and the city is large."), specials=["the", "capital"]) == []


def test_a_finding_detail_is_cut_to_200_characters():
    import hashlib
    long_token = "<" + "".join(hashlib.sha256(str(i).encode()).hexdigest() for i in range(8)) + ">"   # long, and no repeated unit
    found = ps.findings(entry(content=f"Paris {long_token}"), prompt(), specials=[long_token])
    assert [f["kind"] for f in found] == ["control_token_leak"] and len(found[0]["detail"]) <= 200


def test_hiragana_alone_is_recognised_as_cjk():
    assert kinds(entry(content="ここはとてもきれいなまちです。"), prompt(expect_script="CJK")) == []
