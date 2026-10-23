# Microphone side live: `Me` (current model)

- Date: 2026-10-23 ~22:53, MacBook Pro M1 8 GB, built-in microphone.
- Command: `python -m livetr --mic "MacBook"` (both sources open; nothing played on the system side).
- Speaker: the author, spontaneous colloquial Eastern Armenian, ~1.5 min.
- Result: 6 phrases, all labelled `Me`, latency 0.8–1.5 s. Armenian recognised well, including colloquial forms
  (`էս`, `ա`). Translation breaks on colloquial phrases: `ինչ կա, չկա` ("what's up") → "there is no such thing",
  `չատերը` ("the chats") → "chessboard".
- Below: the terminal output as pasted by the author; NeMo's start-up log lines (configuration warnings) are left out.

```text
audio  mic: [3] MacBook Pro Microphone (1 ch, 48000 Hz)
audio  system: [0] BlackHole 2ch (2 ch, 48000 Hz)
loading ASR 'nemo-hy' (nvidia/stt_hy_fastconformer_hybrid_large_pc) on cpu …
loading NLLB translator …
warming up …
ready in 12s — listening. Ctrl-C to stop.

[00:00:03] Me  (1.2s · wait 0.5 · asr 0.2 · spk 0.0 · mt 0.4)
HY: Բարեւ, ինչպես ես ինչ կա, չկա։
EN: Hello, as I am, there is no such thing.

[00:00:17] Me  (0.8s · wait 0.5 · asr 0.1 · spk 0.0 · mt 0.2)
HY: Ի՞նչ ես անում։
EN: What are you doing?

[00:00:28] Me  (1.3s · wait 0.5 · asr 0.2 · spk 0.0 · mt 0.6)
HY: Ի՞նչ կարելի է անել էս դեպքում, եթե հայերենի հիման վրա չատերը նորմալ չի աշխատում։
EN: What can be done if the Armenian-based chessboard is not working properly?

[00:00:58] Me  (1.5s · wait 0.0 · asr 0.2 · spk 0.0 · mt 1.2 · cut)
HY: Եղանակը էսօր մի քիչ ցրտոտ ա, բայց շատ լավ ա մազերս ներկեցի, հետագայում չհասցրեցի մազերս ֆեն, վաղը պետք է արդի։
EN: It's a bit cold today, but I dyed my hair really well, and I didn't get my hair back, so I'll have to get my hair done tomorrow.

[00:01:06] Me  (1.2s · wait 0.5 · asr 0.2 · spk 0.0 · mt 0.5)
HY: Բնական կենտրոն գնամ, տեսնեմ՝ ինչ է իրենից ներկայացնում այդ ամեն ինչը։
EN: I'll go to the nature center, see what all this is about.

[00:01:25] Me  (1.5s · wait 0.5 · asr 0.2 · spk 0.0 · mt 0.8)
HY: Պետական կենտրոն չէ, այլ սիլիկոնի կենտրոնները, արդի ուսումնական կենտրոն է ինքը։
EN: It's not a state center, it's a silicon center, it's an educational center.
```
