# Live test 1 — YouTube conversation through BlackHole

- Date: 2026-10-23, ~15:54, MacBook Pro M1 8 GB, on battery.
- Command: `python -m livetr --no-mic --no-color 2>&1 | tee -i eval/data/live_yt1.txt`
- Input: an Armenian YouTube conversation (2 speakers) played to the Multi-Output Device; call side only.
- ASR model at that time: **`turbo-hy`** (Chillarmo/whisper-large-v3-turbo-armenian). It was replaced afterwards
  by the NVIDIA FastConformer (see DECISIONS.md), so this output shows the *old* model's quality.
- Below is the program's terminal output. Left out: start-up loading bars and a transformers warning that was
  printed once per phrase (68 lines in total; the warning was fixed in commit 11e2a59).
  Nothing else was changed.

```text
audio  system: [0] BlackHole 2ch (2 ch, 48000 Hz)
ready in 18s — listening. Ctrl-C to stop.
[00:00:00] Speaker 1  (4.2s · wait 0.0 · asr 2.9 · spk 0.1 · mt 1.1 · cut)
HY: Ասում եմ, որ իմ «անձնական հետազոտությունը», ո՞ր են կատարել եց ինբոլտակը, աստվելու՝ իր անվորդակներից։
EN: So I'm saying, my personal research, which one of the names of the goddess, the goddess, did the involtation?

[00:00:08] Speaker 1  (3.2s · wait 0.0 · asr 2.4 · spk 0.1 · mt 0.7 · cut)
HY: Ունդական գողջ կյանքի ընթացքում վկայություն ենտարկույթ կամանան ցինը սունին նտոկոսը։
EN: During their lifetime, they have witnessed the death of their own son, Ntozake Shange.

[00:00:16] Speaker 1  (2.9s · wait 0.0 · asr 2.2 · spk 0.1 · mt 0.6 · cut)
HY: Ազոտությունը դա է ցույց տալիս.Ացքն բոլոր տղամարդիկ։
EN: That's what sexuality is all about.

[00:00:24] Speaker 1  (3.8s · wait 0.0 · asr 2.7 · spk 0.1 · mt 1.0 · cut)
HY: Պարդկանց ինըս ու իմ նտոկոսը նույնպես ենթարակվում Է բռնության, և ի իրնչո՞ւ եմ, ե խոզուհմ՝ մշակուկունդի մասին,
EN: Nine of the pigs and my nipple are also subjected to violence, and what am I, a pig, about cultivating?

[00:00:32] Speaker 1  (4.4s · wait 0.0 · asr 3.1 · spk 0.2 · mt 1.2 · cut)
HY: բռնության մշակույթն փոխելու մասին, ընտանիքի մոդելը ֆոլելով մեծ մեր ըտտնիցները, միր մր շատումթերուկ դա բժշտել է։
EN: We're talking about changing the culture of violence, the family model, following our big dreams, and that's what we're doing.

[00:00:40] Speaker 1  (3.0s · wait 0.5 · asr 2.0 · spk 0.1 · mt 0.4)
HY: Մռնության մշակույթն է և այնտագործում է։
EN: It's a culture of death, and it works.

[00:00:44] Speaker 1  (4.1s · wait 0.1 · asr 2.8 · spk 0.1 · mt 1.0 · cut)
HY: Մեր միջավայրը բռնության մջջաավյր է, իսկ մեջա�ավարդը՝ շատ հզոր տարածույթիւն է՝որյա ցանցարին մաստականում է։
EN: Our environment is a hotbed of violence, and the media, a very powerful sphere of influence, is a network of networks.

[00:00:52] Speaker 1  (3.9s · wait 0.0 · asr 2.7 · spk 0.1 · mt 1.0 · cut)
HY: Ցանցի մեջ է վերցնում բոլորին և հասնուամ է՝ բալոտին, ց պարտադրող երականգանակները։
EN: It takes everyone in the net, and I'm going to get the ballot, the force muscles.

[00:01:00] Speaker 1  (4.2s · wait 0.0 · asr 2.9 · spk 0.2 · mt 1.0 · cut)
HY: Եվ շատ քչերն են կարողանում դուրս պրծնել էդ պարտադրանքից, էրդմիջավայրիք էնքնական համանակի։
EN: And very few of us are able to get out of our duties, out of our own self-centeredness.

[00:01:08] Speaker 1  (4.6s · wait 0.0 · asr 3.1 · spk 0.1 · mt 1.4 · cut)
HY: Չերք չէր նել, որ կաղանում են դուրս պրծնել. իրենք, ստեղծության փոքրիկ եթամշակույթ, մեկտոկոսը գուց է, տվյալ դեպք։
EN: It's not like they're trying to scratch the surface. They're a little bit of a creation culture, one-to-many, in this case.

[00:01:16] Speaker 1  (4.7s · wait 0.0 · asr 3.1 · spk 0.2 · mt 1.4 · cut)
HY: Տարբեր իրավիճակներում է, այդպես տարնբել ենթամշակույնե կան, հիմա պաշտոնապեռ ճանաչված եկ՝ եենդամաշատություն մենք չունենից պռնուտի։
EN: It's in different situations, so there's a sub-culture, now we're officially recognized, and we don't have the diversity we need.

[00:01:24] Speaker 1  (4.4s · wait 0.0 · asr 2.8 · spk 0.2 · mt 1.3 · cut)
HY: Այս զերծ, բայց, օքե, թող մէկ տիւ կոսն էլ համարուի, որ դուրսեն պրձել է միջավայրից. բաց մեջածայը շատ կարևոր է, մայնք մոլ։
EN: This is ridiculous, but, well, let's just say that one of the cows is out of the environment.

[00:01:32] Speaker 1  (4.4s · wait 0.0 · asr 3.4 · spk 0.2 · mt 0.8 · cut)
HY: Մշակույթը շատ կարևոր է և ինքը, ընդգրկուն է բոլորին. հետևաբար, բռնութեան մշկակայտով ապրում են։
EN: Culture is very important, and it's inclusive, so we live in a cycle of violence.

[00:01:40] Speaker 1  (4.3s · wait 0.0 · asr 2.8 · spk 0.2 · mt 1.4 · cut)
HY: Մենք բոլորս և չենկ կարող զերծ լինել դրանից, որոնհետև, դա այն մոդելն է, ընտանքի ձևավորուման ըտտնիքներին անդանգամամբնեց։
EN: We can all be free of it, and so, that's the model, that's the model, that's the model that shaped choice.

[00:01:48] Speaker 1  (3.7s · wait 0.0 · asr 2.5 · spk 0.2 · mt 1.0 · cut)
HY: Մանիքի գոյության մոդելն է, որ մենք ժառանգել են։ ով մեզ դաստիարակելեն, տատիկներն եմ դարդիրագետ։
EN: Maniki's model of existence is what we inherited. The grandmothers who raised us are the great-grandmothers.

[00:01:56] Speaker 1  (3.7s · wait 0.0 · asr 2.7 · spk 0.1 · mt 0.9 · cut)
HY: Շ շարունակում դաստիարակել են, պես որ ես չեմ ակնկալուլ, ո՝ իմ, անսաց դոկոսահարցման արդյունք ուներին։
EN: They continue to raise, as I don't expect, the results of my unrelenting docuseries.

[00:02:04] Speaker 1  (3.6s · wait 0.0 · asr 2.6 · spk 0.1 · mt 0.8 · cut)
HY: կը հաստատուի, գուցե մի քիչ աւելի պակաս լինի, Բայց պտկերէ այդպիսինը։
EN: It's going to be fixed, maybe a little less, but it's going to be fixed.

[00:02:12] Speaker 2  (3.8s · wait 0.0 · asr 2.7 · spk 0.2 · mt 0.9 · cut)
HY: չափազանցված չի՞ ձեր պատկերը, ինչո՞ւ, որոյտև մենք հիմա նաև «տեսնում ենց հսկայական փոփուխություն» կահանանձ։
EN: Isn't your image exaggerated, why, even though we're seeing a huge amount of luxury right now?

[00:02:21] Speaker 2  (4.4s · wait 2.0 · asr 2.0 · spk 0.0 · mt 0.4)
HY: - Նկատմամբ կանայք…
EN: - By the way, women...

[00:02:23] Speaker 2  (4.7s · wait 2.6 · asr 1.7 · spk 0.0 · mt 0.3)
HY: Ծխվում են փողոցույք։
EN: They smoke in the streets.

[00:02:25] Speaker 2  (4.3s · wait 0.0 · asr 2.8 · spk 0.2 · mt 1.4 · cut)
HY: Ավտոմեքենայի են վարում, աշխատություն եմն, ընդ որով, երբեմն արխտուած եւ նաեւ, ինչէ եև եուլբեն, «հաճախ տեղեր» ոը աառաջ զուտ մել է։
EN: They drive a car, they work, and sometimes they're unpaved and, even if they're unpaved, they're often the first to get a clean bill.

[00:02:33] Speaker 2  (4.5s · wait 0.0 · asr 3.4 · spk 0.2 · mt 0.9 · cut)
HY: տղամարդկանց գործեր, նրան քաղաքափակայությամբ են զբաղվում, պաշտոններ ունեն, բիզնեսով եմ զաբանվան զիբեղծուկ, և ընթապեսնեցնե ները միստանակ
EN: Men's affairs, urbanism, jobs, business, and a lot of things.

[00:02:41] Speaker 2  (4.5s · wait 0.0 · asr 2.9 · spk 0.2 · mt 1.4 · cut)
HY: Նենց ազատ էին հագնվում ման գալի քաղաքուվ, աառաջ էտ անէլ ինչը չկար, հչքն չ֊կայ փոփուխություն դրան։
EN: They were dressed up like the city of Galilee, and they didn't want to eat anything that wasn't there, and they didn't want to look like it.

[00:02:49] Speaker 1  (2.7s · wait 0.0 · asr 2.1 · spk 0.1 · mt 0.5 · cut)
HY: Փաստերն ակնհայտ են, բոլոր իս անչքի աառաջէն տեսանելի եկը։
EN: The facts are obvious, all of us are in the foreground.

[00:02:57] Speaker 1  (4.2s · wait 0.5 · asr 2.9 · spk 0.1 · mt 0.6)
HY: Վիճարկման ենթակա չեն, ինչպե՞ս, նույն տրամաբանուցյամբ, բոլորիս աչքի աառաջ և տեսանելի է բռնությունը։
EN: How, in the same logic, is violence visible to all of us?

[00:03:04] Speaker 1  (4.4s · wait 0.0 · asr 3.1 · spk 0.2 · mt 1.0 · cut)
HY: Տարբերությունը երկու եևույթների և հարցի ձևակերպման մեջ տարեբեցումը, նրանու՞ւմ է, որ է՛ն եկրևը՝ համարուն։
EN: So the difference between the two verbs and the wording of the question is that it's a number.

[00:03:12] Speaker 1  (4.2s · wait 0.0 · asr 3.0 · spk 0.2 · mt 1.0 · cut)
HY: Որ տեսանելի է և բոլորս գիտենք դրա մասին, բալոսը հպարտորեն նշում ենց դյրար մեսի՜ն ցևչեն�ք «թաքցնույ» երկրորն դար։
EN: As we all know, the Balloon is proud to celebrate the end of the Dark Ages.

[00:03:20] Speaker 1  (3.9s · wait 0.0 · asr 2.7 · spk 0.2 · mt 1.1 · cut)
HY: Որ նոյն չափ տեսանելի է և նույս չապ կասկածել չէ Եվ վիճարկման տեղ չի տալիս, թաքցնում ենք այց։
EN: It's just as visible and it doesn't hesitate and it doesn't cause any controversy, so we hide the visit.

[00:03:28] Speaker 1  (3.9s · wait 0.0 · asr 2.8 · spk 0.1 · mt 0.9 · cut)
HY: Այդ երևույցը, բայք, շատ կարքոր է խոսել, էս թեմայի մասին, ոշած կառևորէ հասկանալ ինչէ՝ բռնութիւնը առ հսարակ։
EN: This is a very short talk, sir, and it's very difficult to understand what this is all about: violence in general.

[00:03:36] Speaker 1  (3.4s · wait 0.0 · asr 2.4 · spk 0.2 · mt 0.8 · cut)
HY: Որովհետև, այո, կանայենք ձեր նշած բոլոր երևույթները կա մեր «միջավայրում» ա յսպես։
EN: Because, yes, let's do all the things you've mentioned that are in our environment.


stopping…

── session stats ──
system  n=30   latency p50 4.16s  p90 4.51s  max 4.69s  over 5s: 0
stage asr   mean 2.70s  max 3.43s
stage mt    mean 0.94s  max 1.42s
stage spk   mean 0.15s  max 0.21s
stage wait  mean 0.21s  max 2.63s
dropped short segments: 1   empty ASR results: 0   audio overflows: 0
```
