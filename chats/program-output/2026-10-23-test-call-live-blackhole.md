# Test call, live through BlackHole (current model)

- Date: 2026-10-23, MacBook Pro M1 8 GB, on battery, no other heavy jobs running.
- Input: `eval/data/test_call_opus16k.wav` (built by `eval/make_test_call.py`): 3 FLEURS voices (F1, F2 = women,
  M1 = man), 9 turns, sent through the 16 kbps Opus codec, then **played with `afplay` to the Multi-Output
  Device**, so the program heard it through BlackHole exactly like a call.
- Command: `python -m livetr --no-mic --no-color` (ASR `nemo-hy` on CPU, NLLB-600M, speaker threshold 0.45).
- True speakers per turn: F1, M1, F2, M1, F1, F2, M1, F2, F1 → printed Speaker 1, 2, 3, 2, 1, 3, 2, 3, 1.
- Score (`eval/score_call.py --offset 1.0`): speaker turns 9/9, 3 of 3 voices, CER 5.8 %, WER 12.6 %,
  latency p50 1.30 s, max 1.59 s.
- Below: the program's terminal output; only NeMo's start-up log lines are left out.

```text
audio  system: [0] BlackHole 2ch (2 ch, 48000 Hz)
loading ASR 'nemo-hy' (nvidia/stt_hy_fastconformer_hybrid_large_pc) on cpu …
OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
No exporters were provided. This means that no telemetry data will be collected.
    {'dir': '???', 'type': 'bpe', 'model_path': 'nemo:56c7ec45a6a84b258c1d87417ed8672e_tokenizer.model', 'vocab_path': 'nemo:a66c4311cce54386973571e261fd4b32_vocab.txt', 'spe_tokenizer_vocab': 'nemo:9482a6c252f94776b9e1b86bbca8963f_tokenizer.vocab'}
     Reason: Missing mandatory value: tokenizer.dir
        full_key: tokenizer.dir
        object_type=dict.
    {'dir': '???', 'type': 'bpe', 'model_path': 'nemo:56c7ec45a6a84b258c1d87417ed8672e_tokenizer.model', 'vocab_path': 'nemo:a66c4311cce54386973571e261fd4b32_vocab.txt', 'spe_tokenizer_vocab': 'nemo:9482a6c252f94776b9e1b86bbca8963f_tokenizer.vocab'}
     Reason: Missing mandatory value: tokenizer.dir
        full_key: tokenizer.dir
        object_type=dict.
    {'dir': '???', 'type': 'bpe', 'model_path': 'nemo:56c7ec45a6a84b258c1d87417ed8672e_tokenizer.model', 'vocab_path': 'nemo:a66c4311cce54386973571e261fd4b32_vocab.txt', 'spe_tokenizer_vocab': 'nemo:9482a6c252f94776b9e1b86bbca8963f_tokenizer.vocab'}
     Reason: Missing mandatory value: tokenizer.dir
        full_key: tokenizer.dir
        object_type=dict.
    {'dir': '???', 'type': 'bpe', 'model_path': 'nemo:56c7ec45a6a84b258c1d87417ed8672e_tokenizer.model', 'vocab_path': 'nemo:a66c4311cce54386973571e261fd4b32_vocab.txt', 'spe_tokenizer_vocab': 'nemo:9482a6c252f94776b9e1b86bbca8963f_tokenizer.vocab'}
     Reason: Missing mandatory value: tokenizer.dir
        full_key: tokenizer.dir
        object_type=dict.
    Train config : 
    manifest_filepath:
    - ???
    - ???
    sample_rate: 16000
    batch_size: 16
    shuffle: true
    num_workers: 8
    pin_memory: true
    max_duration: 20
    min_duration: 0.1
    is_tarred: false
    tarred_audio_filepaths: null
    shuffle_n: 2048
    bucketing_strategy: synced_randomized
    bucketing_batch_size: null
    
    Validation config : 
    manifest_filepath:
    - ???
    - ???
    sample_rate: 16000
    batch_size: 32
    shuffle: false
    use_start_end_token: false
    num_workers: 8
    pin_memory: true
    
    Test config : 
    manifest_filepath: null
    sample_rate: 16000
    batch_size: 32
    shuffle: false
    use_start_end_token: false
    num_workers: 8
    pin_memory: true
    
      super().__init__("LSTM", *args, **kwargs)
    
loading NLLB translator …
warming up …
ready in 11s — listening. Ctrl-C to stop.

[00:00:03] Speaker 1  (1.4s · wait 0.0 · asr 0.3 · spk 0.1 · mt 1.0 · cut)
HY: Մյուս կողմից՝ սառույցի եւ ձյան պայմանները բնական են շատ երկրներում եւ ամբողջ տարին երթեւեկությունը շարունակվում է հիմնականում անընդհատ։
EN: On the other hand, freezing and snowy conditions are normal in many countries, and traffic continues almost continuously throughout the year.

[00:00:11] Speaker ?  (1.1s · wait 0.7 · asr 0.1 · spk 0.0 · mt 0.2)
HY: Հեպով։
EN: It's a novel.

[00:00:14] Speaker 2  (1.6s · wait 0.5 · asr 0.3 · spk 0.1 · mt 0.7)
HY: Այն մեզ տվել է գնացքը ավտոմեքենան եւ շատ այլ փոխադրման սարքեր։
EN: It gave us the train, the car, and many other means of transportation.

[00:00:21] Speaker 3  (1.3s · wait 0.0 · asr 0.3 · spk 0.2 · mt 0.8 · cut)
HY: Միացյալ Նահանգների Փրինսթոն համալսարանի եւ Շեդիայում գտնվող Ուփսալա համալսարանի հետազոտողները զեկուցեցին, որ նոր տեսակ։
EN: Researchers at Princeton University and Uppsala University in the United States have reported a new species.

[00:00:29] Speaker 3  (1.1s · wait 0.0 · asr 0.2 · spk 0.1 · mt 0.8 · cut)
HY: Է ընդամենը երկու սերնդի ընթացքում էվոլյուցիայի են ենթարկվել, չնայած ենթադրվում է, որ այս գործընթացը շատ ավելի երկար է տեւել։
EN: It's only evolved in two generations, although it's thought to have taken much longer.

[00:00:37] Speaker 3  (1.4s · wait 0.0 · asr 0.2 · spk 0.1 · mt 1.0 · cut)
HY: Շնորհիվ Դարբինի էնդեմիկ Սերինոսի, միջին չափսի հողային սերինոսի եւ չվող ակտու սերինոսի։
EN: Thanks to the Darwin endemic serene, the medium-sized terrestrial serene, and the unbreakable acute serene.

[00:00:45] Speaker 3  (1.2s · wait 0.5 · asr 0.2 · spk 0.1 · mt 0.5)
HY: Հինոսկոնիժ տրիսի միջեւ տեղի ունեցած բազմացման։
EN: The hynosconic multiplication between the trii.

[00:00:52] Speaker 2  (1.4s · wait 0.0 · asr 0.3 · spk 0.2 · mt 1.0 · cut)
HY: Ոստիկանության վերակացու Չանդրաշեի քար Սոլանկին, ասաց, որ մեղադրյալները դատարան էին եկել դեմքը ծածկած վ։
EN: Chandrasekhar Stone Solanki, the police chief, said the defendants had come to court face down.

[00:01:03] Speaker 1  (1.2s · wait 0.0 · asr 0.2 · spk 0.1 · mt 0.9 · cut)
HY: Քաղաքականություն բառը ծագել է լատիներեն սիֆիլիս բառից, որը նշանակում է քաղաքակիրթ, որ կապված է լատիներեն սիվիզ բառի։
EN: The word politics comes from the Latin syphilis, meaning civilization, which is related to the Latin word civilization.

[00:01:11] Speaker 1  (1.6s · wait 0.0 · asr 0.3 · spk 0.1 · mt 1.1 · cut)
HY: Հետ ինչ նշանակում է քաղաքացի եւ Սիվիտաս բառի հետ, որը նշանակում է քաղաք կամ քաղաք նահանգ եւ դա նույնպես ինչ֊որ կերպ։
EN: So what does it mean with the word citizen and the word civitas, which means city or city state, and that's also in some way.

[00:01:19] Speaker 1  (1.0s · wait 0.5 · asr 0.1 · spk 0.0 · mt 0.3)
HY: Համանում է սահասարակության ծավալները։
EN: It compares the scope of society.

[00:01:24] Speaker 3  (1.2s · wait 0.0 · asr 0.2 · spk 0.1 · mt 0.8 · cut)
HY: Հիմնականում նա Ուելսը, ի սկզբանե մեզ ստում էր՝ նախ գործելով կարծես թե իրավարական նպատակներից ելնելով՝ երկ։
EN: Basically, he was a Welsh, initially lying to us, acting at first as if for realistic purposes, two.

[00:01:32] Speaker 3  (1.5s · wait 0.5 · asr 0.2 · spk 0.1 · mt 0.7)
HY: Չորրորդ հերթին՝ ձեւացնելով թե՛ մեզ լսում է մինչեւ իր արվեստի գործի ջնջումը։
EN: Fourth, by pretending to listen to us until his work of art is destroyed.

[00:01:39] Speaker 2  (1.2s · wait 0.0 · asr 0.3 · spk 0.1 · mt 0.8 · cut)
HY: Մուհամմադը խորապես հետաքրքրված էր այս աշխարհիկ կյանքից դուրս գտնվող հարցերով։ Խորհելու համար նա հաճախ էր այցելում մի։
EN: Muhammad was deeply interested in matters outside of this worldly life.

[00:01:47] Speaker 2  (1.3s · wait 0.5 · asr 0.2 · spk 0.1 · mt 0.6)
HY: Անցավ, որ հայտնի դարձավ որպես Հիրա Նուռ լույս լեռան վրա։
EN: It went on to become known as the Hira Nur Light on the mountain.

[00:01:54] Speaker 3  (1.4s · wait 0.0 · asr 0.2 · spk 0.1 · mt 1.0 · cut)
HY: Արբանյակն ուղարկվել է տիեզերք հրթիռի միջոցով, տիեզերքում գիտնականները կիրառում են աստղադիտակներ, քանի որ երկրի մթնոլորտն աղավարտվում։
EN: The satellite was launched by a rocket into space, and in space scientists are using telescopes because the Earth's atmosphere is being disrupted.

[00:02:02] Speaker 3  (1.2s · wait 0.5 · asr 0.1 · spk 0.0 · mt 0.5)
HY: Համ է մեր լույսի եւ տեսարանի մի մասը։
EN: It's part of our light and our landscape.

[00:02:08] Speaker 1  (1.4s · wait 0.0 · asr 0.3 · spk 0.1 · mt 1.0 · cut)
HY: Դրա երկար ծնոտներում գտնվում էին ածելիի նման սուր ավելի քան յոթանասունատամ՝ իր բերանի վերին հատվածում ունենալով։
EN: His long knees had more than seventy-seven blades, like those of an axe, at the top of his mouth.

[00:02:17] Speaker 1  (1.4s · wait 0.5 · asr 0.2 · spk 0.1 · mt 0.6)
HY: Եվս մի լրացուցիչ շարք՝ նշանակելով, որ իր ճանապարհը կտրող ցանկացած բան չէր կարող փախչել դրանից։
EN: Another additional line, pointing out that anything that cut its way could not escape it.

stopping…

── session stats ──
system  n=19   latency p50 1.30s  p90 1.55s  max 1.59s  over 5s: 0
stage asr   mean 0.22s  max 0.31s
stage mt    mean 0.75s  max 1.14s
stage spk   mean 0.10s  max 0.19s
stage wait  mean 0.25s  max 0.72s
dropped short segments: 1   empty ASR results: 0   audio overflows: 0
```
