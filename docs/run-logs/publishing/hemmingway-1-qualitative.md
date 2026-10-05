# Hemmingway-1 qualitative check (2-chip p300 package, measured 2026-10-04)

Model purpose: everyday messages, emails and short writing, with no preamble or options (from the Hub card). The prompts below are ones I wrote for that purpose; there are no reference answers.

Settings: package hemmingway-1-p300 server on port 48271, `/v1/chat/completions`, greedy (temperature 0), `enable_thinking: false`, max_tokens 600. Raw output: `raw/p300_qual.json`, script `scripts/qual.py`. Grades are my own judgement of one greedy answer each; they are not a benchmark score and no base-model comparison was run.

Tally: 7 good, 3 partly, 0 poor out of 10.

## 1. What is 7 times 6?

Finish: stop, 3 tokens, 0.3 s wall.

```text
42
```

**Grade: good.** Correct, one line.

## 2. Write the text I send my landlord about the broken boiler. It has been out for four days and it is cold.

Finish: stop, 54 tokens, 1.7 s wall.

```text
Hi, I'm writing to let you know that the boiler has been out for four days now. It's getting really cold in the flat and I'm struggling to keep warm. Could you please arrange for someone to come and look at it as soon as possible?
```

**Grade: good.** Gives the text with no preamble or options. It says what happened and asks for a repair. It does not mention that the flat is cold in winter or ask for a date.

## 3. Write an email to my manager asking for a raise. I have been here two years and took over the billing system last spring.

Finish: stop, 80 tokens, 2.2 s wall.

```text
Subject: Compensation Review Request

Hi [Manager's Name],

I'd like to request a compensation review for my position. I've been with the company for two years now, and since last spring I've taken on additional responsibility by managing the billing system.

I believe my current salary no longer reflects the scope of my role, and I'd appreciate the opportunity to discuss this with you.
```

**Grade: partly.** Gives the email directly, but leaves a literal placeholder '[Manager's Name]', has no sign-off, and does not name the billing-system work as a result.

## 4. Write a message declining my friend's wedding invitation. I can't afford the travel, and I want her to know I still care.

Finish: stop, 73 tokens, 2.3 s wall.

```text
Hi, I'm so happy to hear about your wedding! I'm really glad you're getting married. Unfortunately, I won't be able to make it. I can't afford the travel right now, and I don't want to pretend otherwise. I'm so sorry to miss it. I care about you a lot and I'm rooting for you both.
```

**Grade: good.** Gives the text directly, says it is about travel cost, and says she is cared about. The first two sentences repeat each other.

## 5. My coworker keeps taking credit for my work in meetings. Write the message I send her.

Finish: stop, 83 tokens, 2.4 s wall.

```text
Hi Sarah,

I wanted to talk to you about something that's been bothering me. In the last few meetings, I've noticed that you've been presenting my work as your own. I know it's probably not intentional, but it's important to me that my contributions are recognized properly.

I'd appreciate it if you could make sure to credit me when you reference my work in meetings going forward.
```

**Grade: partly.** Gives the message directly and the tone is calm, but it invents the name 'Sarah' when the prompt gave no name.

## 6. Write a short note to a neighbour whose dog barks all day. I do not want to start a fight.

Finish: stop, 83 tokens, 2.5 s wall.

```text
Hi,

I hope you're doing well. I just wanted to reach out about your dog barking during the day. I know it's probably not something you're aware of, but it's been quite noticeable lately and it's making it a bit hard to get some peace at home.

I'd really appreciate it if you could keep an eye on it when you're able. Thanks so much!
```

**Grade: partly.** Gives the note directly and keeps it friendly, but 'keep an eye on it' does not say what the neighbour should do, and there is no greeting by name or sign-off.

## 7. Write a condolence message to a colleague whose father died this week. I only know him a little.

Finish: stop, 48 tokens, 1.4 s wall.

```text
I'm so sorry to hear about the loss of your father. I know we haven't had much of a chance to get to know each other, but I wanted you to know that I'm thinking of you during this difficult time.
```

**Grade: good.** Short, plain and suitable for someone he barely knows. It does not offer anything specific, which fits the prompt.

## 8. Rewrite this so it sounds like a person wrote it: 'Per my previous correspondence, I am following up regarding the outstanding invoice, which remains unpaid pursuant to the agreed terms.'

Finish: stop, 24 tokens, 0.8 s wall.

```text
Just following up on the invoice we discussed earlier. It's still outstanding and I'd like to get it settled.
```

**Grade: good.** Plain rewrite in two short sentences. It keeps the meaning (invoice outstanding, wants it settled).

## 9. Write the opening paragraph of a short story about a lighthouse keeper who finds a message in a bottle addressed to him.

Finish: stop, 189 tokens, 7.2 s wall.

```text
The bottle had been sitting in the tide pool for what felt like years, though Arthur knew it had been there no more than a week, because he checked the tide pool every morning with the same diligence he checked the lens, the wick, the oil gauge. He found it on a Tuesday in November, wedged beneath a barnacle-crusted rock, its glass clouded with salt and the green film of algae. He pulled it free with his oil-stained fingers and shook it, and something inside rattled with a dry, papery sound. The cork was waxed shut, and around the neck was tied a strip of canvas so old it had gone the color of weak tea. He carried it up the spiral stairs to his kitchen, set it on the table beside the cold coffee and the half-eaten tin of sardines, and spent twenty minutes trying to get the wax to give way without breaking the glass.
```

**Grade: good.** A specific, well-formed opening paragraph with concrete detail. It names the keeper 'Arthur', which the prompt did not give, and does not yet use the bottle's message.

## 10. Write a text to my sister talking her into coming to dad's 70th birthday. She is angry with him and says she won't go.

Finish: stop, 331 tokens, 10.6 s wall.

```text
Hey, I know you're not feeling it right now and I'm not going to pretend I don't know why. I'm not going to tell you to just forget about what happened because that's not fair to you.

But I need you to think about this for a second. He's seventy. That's not a small thing. He's going to be seventy years old and he's going to be sitting in that living room with the same old couch and the same old photos on the wall, and he's going to look around and he's going to notice you're not there. He won't say anything about it. He'll just notice.

I know you're angry. I know you have every right to be. But I also know him. He's not going to make it easy for you to come back. He's not going to call you and say "hey, I'm sorry, can we talk?" He's going to act like everything is fine and he's going to hope you show up, and if you don't, he's going to quietly be hurt about it and never mention it. That's just how he is.

I'm not asking you to forgive him. I'm not asking you to pretend it didn't happen. I'm just asking you to be there for a few hours, eat some cake, and let him see your face. That's all. You don't have to have a conversation with him. You don't have to hug him. You can just be in the room.

And honestly, I'd like to see you too. It's been a while.
```

**Grade: good.** Reads like a message between siblings and does not wave away her anger. It is long for a text message and leans on repeated 'I'm not going to / I'm not asking' phrasing.
