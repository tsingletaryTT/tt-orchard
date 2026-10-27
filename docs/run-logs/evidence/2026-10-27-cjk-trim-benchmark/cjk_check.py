"""CJK check for the trimmed model against its base, defined before any run (2026-10-27).

Two commands:
  python3 cjk_check.py run --url http://127.0.0.1:PORT --served NAME --out FILE   # sends the 40 prompts, one at a time
  python3 cjk_check.py compare --trim FILE --base FILE                            # prints the comparison as JSON

The 40 prompts: 10 Chinese, 10 Japanese, 10 Korean, 10 mixed (CJK text inside English or between languages). They are
the kinds of text the removed vocabulary entries stand for: ordinary prose questions, a translation, a transcription of
rare characters, a code comment in Chinese, names, and a request to answer in the asked language. Each request is greedy
(temperature 0), thinking as the chat template defaults it, max_tokens 8192, one request at a time.

Per prompt it records: the reply text, finish reason, completion tokens, and these measures of the reply (content only):
  - script_share: the share of letters in the reply that are Han, Kana or Hangul, by script
  - expected_script_ok: whether the reply uses the script the prompt asks for (Han for Chinese, Kana for Japanese,
    Hangul for Korean, both CJK and Latin for mixed) at a share of at least MIN_SCRIPT_SHARE of its letters
  - well_formed: a non-empty reply, finished with "stop", with no U+FFFD, no 'unk' marker and no run of 6 or more
    identical characters
  - exact_match_with_other: for the comparison, whether the two models gave the same text
and for the comparison: counts per language of each measure, and the agreement between the models on `well_formed` and on
`expected_script_ok`, the number of identical replies, and the number of prompts where only one model failed a measure.
The prompts hold 40 items, so a difference of one or two prompts is not evidence of anything.
"""
import json, re, sys, time, urllib.request, argparse, unicodedata

MIN_SCRIPT_SHARE = 0.3

PROMPTS = [
    # Chinese (zh)
    ("zh", "请用三句话解释什么是光合作用。"),
    ("zh", "把下面这句话翻译成英文：「海内存知己，天涯若比邻。」"),
    ("zh", "请写一首四句的五言绝句，主题是秋天的月亮。"),
    ("zh", "「饕餮」「魑魅魍魉」「龘」这几个字分别是什么意思？请逐个解释。"),
    ("zh", "请用中文介绍一下北京故宫的历史，不超过一百字。"),
    ("zh", "写一个 Python 函数，计算斐波那契数列的第 n 项，并用中文写注释。"),
    ("zh", "请列出中国的四大发明，并各用一句话说明它们的作用。"),
    ("zh", "请纠正这句话的错别字：「我门今天去公圆玩，天器很好。」"),
    ("zh", "请把「繁體字的書寫比簡體字複雜」转换成简体字，再转换回繁体字。"),
    ("zh", "请用文言文写一段话，描述一个人在雨中等朋友。"),
    # Japanese (ja)
    ("ja", "光合成について、三文で説明してください。"),
    ("ja", "次の文を英語に翻訳してください：「吾輩は猫である。名前はまだ無い。」"),
    ("ja", "春の朝をテーマにした俳句を一つ作ってください。"),
    ("ja", "「鬱」「薔薇」「躊躇」の読み方と意味を教えてください。"),
    ("ja", "東京タワーの歴史を百字以内で紹介してください。"),
    ("ja", "Pythonでフィボナッチ数列のn番目を求める関数を書き、日本語でコメントを付けてください。"),
    ("ja", "ひらがなだけで「ありがとうございます」を使った短いお礼の手紙を書いてください。"),
    ("ja", "次の文の誤字を直してください：「私わ昨日、図書館え行きました。」"),
    ("ja", "「今日はいい天気ですね」を、丁寧語、尊敬語、くだけた言い方の三通りで言い換えてください。"),
    ("ja", "カタカナ語「コンピューター」「アルゴリズム」「インターネット」を漢字を使った言葉で説明してください。"),
    # Korean (ko)
    ("ko", "광합성이 무엇인지 세 문장으로 설명해 주세요."),
    ("ko", "다음 문장을 영어로 번역해 주세요: 「가는 말이 고와야 오는 말이 곱다.」"),
    ("ko", "가을 달을 주제로 짧은 시를 한 편 써 주세요."),
    ("ko", "「꿈」「사랑」「그리움」이라는 단어의 뜻과 어원을 설명해 주세요."),
    ("ko", "경복궁의 역사를 백 자 이내로 소개해 주세요."),
    ("ko", "피보나치 수열의 n번째 항을 구하는 파이썬 함수를 쓰고, 한국어로 주석을 달아 주세요."),
    ("ko", "한글의 자음과 모음이 만들어진 원리를 간단히 설명해 주세요."),
    ("ko", "다음 문장의 맞춤법을 고쳐 주세요: 「저는 어제 도서관에 갓다가 친구를 만났읍니다.」"),
    ("ko", "존댓말과 반말의 차이를 예문 두 개씩으로 보여 주세요."),
    ("ko", "한자어 「學校」「圖書館」「大學」의 한국어 발음과 뜻을 알려 주세요."),
    # Mixed (mx): CJK inside English, or several CJK languages in one prompt
    ("mx", "What does the Chinese idiom 画蛇添足 mean, and when would you use it? Answer in English and quote the idiom once."),
    ("mx", "Translate 'The library opens at nine' into Chinese, Japanese and Korean, and give each in its own script."),
    ("mx", "The Japanese word 木漏れ日 has no single English word. Explain what it means in two sentences."),
    ("mx", "Compare how 학교, 学校 and 學校 are written and pronounced in Korean, Japanese and Chinese."),
    ("mx", "Write a Python dict mapping the numbers 1 to 5 to their Chinese characters (一二三四五) and Japanese readings in romaji."),
    ("mx", "Here is a log line: '2026-10-26 ERROR 用户登录失败：密码错误 (user=山田太郎)'. Explain it in English and name the language of each part."),
    ("mx", "Give three common Korean surnames, three Japanese surnames and three Chinese surnames, each with its original characters or Hangul."),
    ("mx", "Explain the difference between 它, 他 and 她 in Chinese in one paragraph of English."),
    ("mx", "Reply with exactly one line: the Japanese greeting for the morning, written in kanji and hiragana, then its English meaning in brackets."),
    ("mx", "User question in Chinese: 我的电脑开机后黑屏，怎么办？ Reply in Chinese with three numbered steps."),
]
assert len(PROMPTS) == 40 and [sum(1 for l, _ in PROMPTS if l == k) for k in ("zh", "ja", "ko", "mx")] == [10] * 4


def script_of(ch):
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "hangul"
    if 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF:
        return "kana"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2A6DF or 0xF900 <= o <= 0xFAFF:
        return "han"
    if unicodedata.category(ch).startswith("L"):
        return "other"
    return None


def measures(lang, text, finish):
    counts = {"han": 0, "kana": 0, "hangul": 0, "other": 0}
    for ch in text:
        s = script_of(ch)
        if s:
            counts[s] += 1
    total = sum(counts.values()) or 1
    share = {k: round(v / total, 4) for k, v in counts.items()}
    cjk = share["han"] + share["kana"] + share["hangul"]
    ok = {"zh": share["han"] >= MIN_SCRIPT_SHARE, "ja": share["kana"] + share["han"] >= MIN_SCRIPT_SHARE and counts["kana"] > 0,
          "ko": share["hangul"] >= MIN_SCRIPT_SHARE, "mx": cjk > 0 and share["other"] > 0}[lang]
    formed = bool(text.strip()) and finish == "stop" and "�" not in text and "unk" not in text.lower().split() \
        and not re.search(r"(.)\1{5,}", text)
    return {"script_share": share, "expected_script_ok": ok, "well_formed": formed}


def post(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.load(r)


def run(a):
    out = []
    for i, (lang, prompt) in enumerate(PROMPTS):
        t = time.time()
        try:
            r = post(a.url, {"model": a.served, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 8192})
            ch = r["choices"][0]
            text = ch["message"].get("content") or ""
            rec = {"i": i, "lang": lang, "prompt": prompt, "reply": text, "finish": ch.get("finish_reason"),
                   "completion_tokens": r["usage"]["completion_tokens"], "prompt_tokens": r["usage"]["prompt_tokens"],
                   "had_reasoning": bool(ch["message"].get("reasoning_content") or ch["message"].get("reasoning"))}
            rec.update(measures(lang, text, ch.get("finish_reason")))
        except Exception as e:  # a failed request is a record, not a stop
            rec = {"i": i, "lang": lang, "prompt": prompt, "error": repr(e)[:300], "well_formed": False, "expected_script_ok": False}
        rec["wall_s"] = round(time.time() - t, 1)
        out.append(rec)
        print(i, lang, rec.get("finish"), rec.get("completion_tokens"), rec["well_formed"], rec["expected_script_ok"], flush=True)
    json.dump({"prompts": out, "min_script_share": MIN_SCRIPT_SHARE}, open(a.out, "w"), ensure_ascii=False, indent=1)


def compare(a):
    t = {r["i"]: r for r in json.load(open(a.trim))["prompts"]}
    b = {r["i"]: r for r in json.load(open(a.base))["prompts"]}
    res = {"n": len(t), "by_language": {}}
    for lang in ("zh", "ja", "ko", "mx"):
        ids = [i for i in t if t[i]["lang"] == lang]
        d = {"n": len(ids)}
        for m in ("well_formed", "expected_script_ok"):
            d[m] = {"trim": sum(1 for i in ids if t[i][m]), "base": sum(1 for i in ids if b[i][m]),
                    "only_trim_fails": sum(1 for i in ids if not t[i][m] and b[i][m]), "only_base_fails": sum(1 for i in ids if t[i][m] and not b[i][m])}
        d["identical_replies"] = sum(1 for i in ids if t[i].get("reply") is not None and t[i].get("reply") == b[i].get("reply"))
        d["tokens_median"] = {k: sorted(x[i].get("completion_tokens", 0) for i in ids)[len(ids) // 2] for k, x in (("trim", t), ("base", b))}
        res["by_language"][lang] = d
    res["identical_replies_total"] = sum(v["identical_replies"] for v in res["by_language"].values())
    # first differing position of replies that are not identical, as a share of the shorter reply
    diffs = []
    for i in t:
        x, y = t[i].get("reply") or "", b[i].get("reply") or ""
        if x != y:
            k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
            diffs.append({"i": i, "lang": t[i]["lang"], "first_diff_char": k, "len_trim": len(x), "len_base": len(y)})
    res["differing"] = diffs
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--url", required=True); r.add_argument("--served", required=True); r.add_argument("--out", required=True)
    c = sub.add_parser("compare"); c.add_argument("--trim", required=True); c.add_argument("--base", required=True)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)
