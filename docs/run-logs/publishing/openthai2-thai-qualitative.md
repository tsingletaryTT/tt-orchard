# Thai quality check: iapp/openthai2.0-qwen3.8-27b (p300 package, 2 chips)

Measured 2026-10-04 on the installed p300 package. 10 prompts written for this check, greedy (temperature 0), thinking off (`enable_thinking: false`), max 700 output tokens, one at a time through `/v1/chat/completions`. Raw data: `raw/thai_answers.json`; script: `raw/thai_run.py`; prompts: `raw/thai_prompts.json`.

Grades are one reader's judgment (the agent that ran the check, reading Thai). Where a reference answer could be verified by arithmetic or by comparison with a supplied text it was; no other reference answers were invented. No base-model comparison was run, so nothing here says whether the fine-tune is better or worse than base Qwen3.8-27B.

Tally: 6 correct, 4 partly, 0 wrong, 0 not Thai (T3 is a Thai-to-English translation and is meant to be English).

## T1 (general knowledge)

Prompt:

```
อธิบายว่าทำไมท้องฟ้าในตอนกลางวันจึงมีสีฟ้า ตอบไม่เกิน 4 ประโยค
```

Answer (finish: stop, 194 tokens):

```
แสงแดดประกอบด้วยแสงสีต่างๆ เมื่อเดินทางผ่านบรรยากาศ แสงสีฟ้าซึ่งมีความยาวคลื่นสั้นจะถูกกระเจิงไปทุกทิศทางมากที่สุดโดยโมเลกูลของอากาศ

ดังนั้นเมื่อเราดูขึ้นไปท้องฟ้าจึงเห็นแสงสีฟ้าที่ถูกกระเจิงเข้าสู่ตาของเราเป็นสีหลัก

ส่วนแสงสีแดงหรือสีเหลืองที่ถูกกระเจิงน้อยกว่าจะเดินทางตรงไปยังผ้าดินหรือถึงตาเราในรูปแบบแสงแดดโดยตรง

จึงทำให้ท้องฟ้าในช่วงกลางวันมีลักษณะเป็นสีฟ้าชัดเจน
```

Grade: partly - the physics is right (shorter blue wavelengths scatter most), but one phrase is garbled ("ผ้าดิน") and it uses 4 short paragraphs, not 4 sentences.

## T2 (summarize)

Prompt:

```
สรุปข้อความต่อไปนี้เป็น 2 ประโยค

เมื่อวันเสาร์ที่ผ่านมา ชุมชนริมคลองบางกอกน้อยจัดงานตลาดน้ำประจำปีเป็นครั้งที่ 10 มีพ่อค้าแม่ค้ามากกว่า 80 ร้านนำอาหารพื้นบ้านและของใช้ทำมือมาขายบนเรือและริมตลิ่ง ผู้จัดงานระบุว่าปีนี้มีนักท่องเที่ยวมาเที่ยวประมาณ 5,000 คน มากกว่าปีที่แล้วราวร้อยละ 20 รายได้ส่วนหนึ่งจะนำไปใช้ทำความสะอาดคลองและซ่อมแซมท่าเรือของชุมชน นอกจากนี้ยังมีการแสดงดนตรีไทยและเวิร์กช็อปทำขนมไทยสำหรับเด็ก ๆ ด้วย
```

Answer (finish: stop, 143 tokens):

```
งานตลาดน้ำประจำปีครั้งที่ 10 ของชุมชนริมคลองบางกอกน้อย ดึงดูดนักท่องเที่ยวมาเที่ยวราว 5,000 คน ซึ่งเพิ่มขึ้นจากปีก่อนหน้าประมาณร้อยละ 20

รายได้ส่วนหนึ่งจากงานจะนำไปใช้ในการทำความสะอาดคลองและซ่อมแซมท่าเรือ รวมถึงมีการจัดแสดงดนตรีไทยและเวิร์กช็อปทำขนมไทยให้เด็กๆ ได้ร่วมกิจกรรมด้วย
```

Grade: correct - 2 sentences; all figures (10th year, about 5,000 visitors, up about 20 percent, cleanup and pier repair, Thai music and sweets workshop) match the source paragraph.

## T3 (translate Thai to English)

Prompt:

```
แปลประโยคต่อไปนี้เป็นภาษาอังกฤษ: "ฝนตกหนักมากเมื่อเช้านี้ ฉันเลยต้องเปลี่ยนแผนและอยู่บ้านทำอาหารแทนการไปตลาด"
```

Answer (finish: stop, 27 tokens):

```
It rained heavily this morning, so I had to change my plans and stay home to cook instead of going to the market.
```

Grade: correct - accurate, fluent English.

## T4 (translate English to Thai)

Prompt:

```
Translate into Thai: "The meeting has been moved to Thursday afternoon, so please send me your report by Wednesday evening."
```

Answer (finish: stop, 50 tokens):

```
การประชุมถูกย้ายไปเป็นวันพฤหัสบดีช่วงบ่าย ดังนั้นขอให้ส่งรายงานมาให้ผมก่อนเวลาเย็นของวันอังคาร
```

Grade: partly - fluent Thai, but the deadline is wrong: the source says Wednesday evening, the answer says Tuesday evening (เย็นของวันอังคาร).

## T5 (instruction following)

Prompt:

```
เขียนประโยคภาษาไทย 3 ประโยคเกี่ยวกับการปลูกต้นไม้ โดยแต่ละประโยคต้องขึ้นต้นด้วยคำว่า "วันนี้" และไม่ต้องมีคำอธิบายอื่นเพิ่ม
```

Answer (finish: stop, 65 tokens):

```
วันนี้ผมได้ไปปลูกต้นมะม่วงในหลังบ้าน
วันนี้เราช่วยกันปลูกต้นมะม่วงหาวมะนาวโห่
วันนี้ครูพาเด็กๆ ไปปลูกต้นไม้ในสวนสาธารณะ
```

Grade: partly - three sentences, each begins with "วันนี้", no extra text, as asked; the second sentence contains a nonsense phrase ("ต้นมะม่วงหาวมะนาวโห่").

## T6 (arithmetic word problem)

Prompt:

```
แม่ซื้อมะม่วง 3 กิโลกรัม ราคากิโลกรัมละ 45 บาท และซื้อส้ม 2 กิโลกรัม ราคากิโลกรัมละ 60 บาท แม่จ่ายเงินไป 500 บาท แม่จะได้เงินทอนเท่าไร แสดงวิธีคิดสั้น ๆ
```

Answer (finish: stop, 138 tokens):

```
**วิธีคิด:**
1. คำนวณราคามะม่วง: $3 \times 45 = 135$ บาท
2. คำนวณราคาส้ม: $2 \times 60 = 120$ บาท
3. รวมค่าของทั้งหมด: $135 + 120 = 255$ บาท
4. คำนวณเงินทอน: $500 - 255 = 245$ บาท

**คำตอบ:**
แม่จะได้เงินทอน **245** บาท
```

Grade: correct - 135 + 120 = 255, change 245 baht; checked by hand.

## T7 (JSON extraction)

Prompt:

```
ดึงข้อมูลจากข้อความต่อไปนี้ออกมาเป็น JSON ที่มีคีย์ name, age, city, occupation, phone ตอบเป็น JSON อย่างเดียว

คุณสมชาย ใจดี อายุ 42 ปี อาศัยอยู่ที่จังหวัดเชียงใหม่ ทำงานเป็นวิศวกรโยธา ติดต่อได้ที่เบอร์ 081-234-5678
```

Answer (finish: stop, 56 tokens):

```
{"name": "สมชาย ใจดี", "age": 42, "city": "เชียงใหม่", "occupation": "วิศวกรโยธา", "phone": "081-234-5678"}
```

Grade: correct - valid JSON, all five fields match the text (age is a number, as is natural).

## T8 (general knowledge)

Prompt:

```
เทศกาลสงกรานต์ของไทยมีความสำคัญอย่างไร และคนไทยทำกิจกรรมอะไรกันบ้างในช่วงเทศกาลนี้
```

Answer (finish: length, 700 tokens):

```
**เทศกาลสงกรานต์** เป็นประเพณีปีใหม่ของไทย ซึ่งจัดขึ้นในช่วงวันที่ 13 ถึง 15 เมษายน ของทุกปี โดยมีความสำคัญและกิจกรรมหลักๆ ดังนี้ครับ:

### ความสำคัญของเทศกาลสงกรานต์
1. **เป็นประเพณีปีใหม่ไทย**: สงกรานต์มาจากภาษาสันสกฤต แปลว่า "การเดินขึ้น" หรือ "การเดินเข้า" ซึ่งหมายถึงการที่ดวงอาทิตย์เดินขึ้นสู่ราศีเมษ ทำให้เป็นจุดเริ่มต้นของปีใหม่ในปัญจทาไทย
2. **การเคารพพุทธรัฐและพุทธศาสนา**: เป็นโอกาสสำคัญในการทำบุญ ทำความดี และแสดงความเคารพต่อพระพุทธศาสนา
3. **การเคารพผู้ใหญ่และครอบครัว**: เป็นวันสำคัญในการแสดงความกตัญญูต่อพ่อแม่และผู้ใหญ่ในครอบครัว
4. **การสืบสานวัฒนธรรมและประเพณีไทย**: ช่วยรักษาและสืบทอดประเพณีไทยให้คงอยู่ตลอดไป

### กิจกรรมที่คนไทยทำในช่วงเทศกาลสงกรานต์
1. **การทำบุญต่างๆ**:
   - **ตักบาตร**: ออกตักบาตรให้พระสงฆ์ในเช้า
   - **ทำบุญที่วัด**: ไปทำบุญ ที่วัด ฟังธรรม และทำพุทธกิจ
   - **สรงน้ำพระพุทธรูป**: สรงน้ำพระพุทธรูปเพื่อเป็นการเคารพและเคารพพระพุทธศาสนา

2. **การสรงน้ำพระพุทธรูปและพุทธพิหาร**:
   - สรงน้ำพระพุทธรูปและพุทธพิหารในวัด เพื่อเป็นการเคารพและเคารพพระพุทธศาสนา

3. **การสรงน้ำพระพุทธรูปและพุทธพิหาร**:
   - สรงน้ำพระพุทธรูปและพุทธพิหารในวัด เพื่อเป็นการเคารพและเคารพพระพุทธศาสนา

4. **การสรงน้ำพระพุทธรูปและพุทธพิหาร**:
   - สรงน้ำพระพุทธรูปและพุทธพิหารในวัด เพื่อเป็นการเคารพและเคารพพระพุทธศาสนา

5. **การสรงน้ำพระพุทธรูปและพุทธพิหาร**:
   - สรงน้ำพระพุทธรูปและพุทธพิหารในวัด เพื่อเป็นการเคารพและเคารพพระพุทธศาสนา

6. **การสรงน้ำพระพุทธรูปและพ
```

Grade: partly - fluent and mostly reasonable (13-15 April, merit-making, bathing Buddha images, respect for elders), but contains invented or garbled terms ("ปัญจทาไทย", "พุทธรัฐ"). The answer hit the 700-token cap and is cut off mid-list. The "sun moves into Aries" etymology was not checked against a reference.

## T9 (arithmetic word problem)

Prompt:

```
รถไฟขบวนหนึ่งออกจากกรุงเทพฯ เวลา 08:30 น. ใช้เวลาเดินทาง 7 ชั่วโมง 45 นาที รถไฟจะถึงปลายทางเวลาเท่าไร
```

Answer (finish: stop, 8 tokens):

```
16:15 น.
```

Grade: correct - 08:30 + 7 h 45 min = 16:15; the answer "16:15 น." is right but gives no working.

## T10 (short writing task)

Prompt:

```
เขียนอีเมลสั้น ๆ ถึงหัวหน้า เพื่อขอลาพักร้อน 2 วันในวันที่ 15-16 พฤศจิกายน ใช้ภาษาสุภาพ
```

Answer (finish: stop, 123 tokens):

```
สวัสดีครับ/ค่ะ

ขออภัยที่รบกวนเวลาครับ/ค่ะ ผม/ดิฉันขอแจ้งความต้องการลาพักร้อนจำนวน 2 วัน ในวันที่ 15-16 พฤศจิกายน นี้

หากมีงานที่ต้องเตรียมการล่วงหน้า ผม/ดิฉันจะดำเนินการมอบหมายงานให้พร้อมกันด้วยครับ/ค่ะ

ขอแสดงความเคารพ
[ชื่อของคุณ]
```

Grade: correct - polite, well-formed Thai email asking for leave on 15-16 November; it uses "ครับ/ค่ะ" and "ผม/ดิฉัน" and a name placeholder, which is reasonable because the sender is not specified.
