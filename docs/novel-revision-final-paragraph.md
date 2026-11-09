# Novel revision: rewriting a chapter's final paragraph

SonicBoom's novel-revision workload sends a complete chapter and one editing
instruction as a single chat turn, decodes the reply, and writes the
detokenized reply unmodified. This page records what two quantized models
produce for the same instruction — keep the chapter, replace its final
paragraph with a different ending — and the decode throughput measured in the
same runs. Every run uses the `auto` speculative policy (`SONICBOOM_SPEC=auto`).

The page records one instruction on two chapters. It is not a general
evaluation of either model.

## The request

| lang | source (public domain) | chapter tokens |
|---|---|---|
| en | Pride and Prejudice, chapters I–II | 2,203 |
| zh | 红楼梦, 第三回 (excerpt) | 2,972 |

Instruction 3, English:

> Keep the chapter above exactly as it is except for its final paragraph.
> Replace that last paragraph with a different ending of about the same
> length: same characters and setting, but a different decision and a
> different closing image. Output the complete revised chapter and nothing
> else.

Instruction 3, Chinese:

> 请保留上面章节的前面部分不动，只重写最后一段，给这一回换一个不同的结局，长度与原来大致相当。请输出修改后的完整章节，不要输出其他内容。

The final paragraph the instruction refers to, English (last paragraph of
chapter II):

> The rest of the evening was spent in conjecturing how soon he would return
> Mr. Bennet’s visit, and determining when they should ask him to dinner.

Chinese (the excerpt's last paragraph, from 王熙凤's entrance, 772 characters):

> 一語未了，只听后院中有人笑聲，說：“我來遲了，不曾迎接遠客！"黛玉納罕道：“這些人個個皆斂聲屏气，恭肅嚴整如此，這來者系誰，這樣放誕無禮？"心下想時，只見一群媳婦丫鬟圍擁著一個人從后房門進來．這個人打扮与眾姑娘不同，彩繡輝煌，恍若神妃仙子：頭上戴著金絲八寶攢珠髻，綰著朝陽五鳳挂珠釵，項上戴著赤金盤螭瓔珞圈，裙邊系著豆綠宮絛，雙衡比目玫瑰佩，身上穿著縷金百蝶穿花大紅洋緞窄□襖，外罩五彩刻絲石青銀鼠褂，下著翡翠撒花洋縐裙．一雙丹鳳三角眼，兩彎柳葉吊梢眉，身量苗條，体格風騷，粉面含春威不露，丹唇未起笑先聞．黛玉連忙起身接見．賈母笑道，"你不認得他，他是我們這里有名的一個潑皮破落戶儿，南省俗謂作`辣子'，你只叫他`鳳辣子'就是了。”黛玉正不知以何稱呼，只見眾姊妹都忙告訴他道：“這是璉嫂子。”黛玉雖不識，也曾听見母親說過，大舅賈赦之子賈璉，娶的就是二舅母王氏之內侄女，自幼假充男儿教養的，學名王熙鳳．黛玉忙陪笑見禮，以"嫂"呼之．這熙鳳攜著黛玉的手，上下細細打諒了一回，仍送至賈母身邊坐下，因笑道：“天下真有這樣標致的人物，我今儿才算見了！況且這通身的气派，竟不象老祖宗的外孫女儿，竟是個嫡親的孫女，怨不得老祖宗天天口頭心頭一時不忘．只可怜我這妹妹這樣命苦，怎么姑媽偏就去世了！"說著，便用帕拭淚．賈母笑道：“我才好了，你來招我．你妹妹遠路才來，身子又弱，也才勸住了，快再休提前話。”這熙鳳听了，忙轉悲為喜道：“正是呢！我一見了妹妹，一心都在他身上了，又是喜歡，又是傷心，竟忘記了老祖宗．該打，該打！"又忙攜黛玉之手，問：“妹妹几歲了？可也上過學？現吃什么藥？在這里不要想家，想要什么吃的，什么玩的，只管告訴我，丫頭老婆們不好了，也只管告訴我。”一面又問婆子們：“林姑娘的行李東西可搬進來了？帶了几個人來？你們赶早打掃兩間下房，讓他們去歇歇。”

## Run setup

- Each run is a fresh session: the prompt is the chapter plus the instruction,
  placed in the model's chat template as one user turn. The prompt-token
  counts below include the template and the instruction.
- Greedy: temperature 0, top-p 1.0, seed 0, three runs per cell.
- Sampled: temperature 0.7, top-p 1.0, seed 0, one run per cell.
- The generation budget is fixed at 1.2× the chapter's token count. A reply is
  not cut at the model's closing sentence; generation continues until the
  budget is used. All sixteen runs below used the full budget.
- `decode tok/s` is (generated tokens − 1) divided by the time after the first
  token. `TTFT` is the full-prompt prefill plus the first token. Values are
  medians over the three greedy runs, or the single sampled run.
- The greedy runs include speculative drafting and batched verification (the
  `auto` policy). The sampled runs report zero speculative iterations and are
  plain decode.

## Models, hardware, and runtime

| model | file | quant | size | SHA-256 (first 16) |
|---|---|---|---|---|
| Gemma 4 E4B IT | `gemma-4-E4B-it-Q5_K_M.gguf` | Q5_K_M | 5.10 GiB | `a5d6e634db151368` |
| Qwen2.5-7B-Instruct-1M | `qwen2.5-7b-instruct-1m-q4_k_m.gguf` | Q4_K_M | 4.36 GiB | `6dba997502a27b5f` |

RTX 3050 (8 GiB, sm_86, driver 595.84), Ryzen 5 5500, Ubuntu 24.04, g++ 13.3.
SonicBoom at commit `f6e620d`, CUDA spine, model resident on the GPU.

## Throughput (auto)

| model | lang | arm | runs | prompt tokens | generated | TTFT (ms) | decode tok/s |
|---|---|---|---|---|---|---|---|
| Gemma 4 E4B Q5_K_M | en | greedy | 3 | 2,311 | 2,697 | 1,839 | 222.9 |
| Gemma 4 E4B Q5_K_M | en | sampled | 1 | 2,311 | 2,697 | 1,842 | 43.8 |
| Gemma 4 E4B Q5_K_M | zh | greedy | 3 | 3,051 | 3,598 | 2,207 | 293.1 |
| Gemma 4 E4B Q5_K_M | zh | sampled | 1 | 3,051 | 3,598 | 2,214 | 43.3 |
| Qwen2.5-7B Q4_K_M | en | greedy | 3 | 2,275 | 2,643 | 2,198 | 99.8 |
| Qwen2.5-7B Q4_K_M | en | sampled | 1 | 2,275 | 2,643 | 2,204 | 37.9 |
| Qwen2.5-7B Q4_K_M | zh | greedy | 3 | 3,033 | 3,566 | 2,763 | 114.1 |
| Qwen2.5-7B Q4_K_M | zh | sampled | 1 | 3,033 | 3,566 | 2,753 | 37.5 |

The greedy replies reproduce most of the chapter, so the drafter predicts most
tokens exactly and the greedy rates include that acceptance. The sampled runs
use the plain decode path and are shown for comparison.

## English chapter

### Gemma 4 E4B Q5_K_M — greedy

The reply reproduces chapters I–II and replaces the final paragraph with:

> The rest of the evening was spent in quiet, feverish planning, with Mrs.
> Bennet dictating a precise schedule of calls and visits, while Mr. Bennet,
> leaning back in his chair, watched the ceiling, a faint, unreadable smile
> playing upon his lips, content to observe the delightful frenzy of his
> domestic life.

The paragraph is followed by one `>` character. The other two greedy runs
produce the same token stream.

### Gemma 4 E4B Q5_K_M — sampled

The reply reproduces chapters I–II and replaces the final paragraph with:

> The rest of the evening was spent in quiet agreement, a low, determined
> murmur among the daughters about the merits of the new arrival. Jane, who
> had been silent until now, finally spoke, her voice soft but clear. “Perhaps
> we should wait until we know more of his character before we allow ourselves
> to be overly fond of the notion,” she suggested gently. Elizabeth, whose
> interest had been piqued by the brief exchange, nodded in agreement. “It
> seems too swift for such a profound matter, Mamma. Let us observe, and see
> where this fortunate acquaintance may lead us.” They all settled into a
> thoughtful silence, the glittering prospect of fortune momentarily
> overshadowed by the quiet dignity of caution.

The reply continues with a run of `>` characters to the token budget.

### Qwen2.5-7B-Instruct-1M Q4_K_M — greedy

The reply begins at chapter II (`Mr. Bennet was among the earliest of those
who waited on Mr. Bingley.`); chapter I and the `CHAPTER II.` heading are not
in the reply. Chapter II is reproduced until the close of the scene, which is
rewritten from Mrs. Bennet's reaction onward:

> The astonishment of the ladies was just what he wished—though perhaps Mrs.
> Bennet’s astonishment was not quite as great as the rest. When the first
> tumult of surprise had passed, she began to reflect on the situation with a
> more measured calm. She realized that this visit, far from being a mere
> social call, could be the beginning of something more substantial. The
> Bennets had always been a family of quiet means, and this opportunity to
> connect with a man of fortune could be a turning point for their daughters.
> She decided that she would not let this chance slip away.
>
> “Very well, my dear,” she said, with a newfound resolve. “If you have
> already paid the visit, then we must make the most of it. I will not neglect
> the opportunity to introduce our daughters to Mr. Bingley. We must be
> present at the assemblies and ensure that our girls are seen in the best
> light. After all, it is not just about the visit; it is about the future of
> our children.”
>
> The rest of the evening was spent in planning how to present their daughters
> at the assemblies and in preparing for the upcoming ball. The Bennets were
> determined to make the most of this chance, and the expectations for the
> future were high.

After the closing sentence the reply continues with a repeated-symbol and
repeated-word sequence (emoji and `citiz`) to the token budget.

### Qwen2.5-7B-Instruct-1M Q4_K_M — sampled

The reply begins with the `CHAPTER II.` heading; chapter I is not in the
reply. Chapter II is reproduced with light rephrasing, and the close of the
scene is rewritten:

> The astonishment of the ladies was just what he wished—though perhaps Mrs.
> Bennet's perhaps surpassed the rest. When the initial surprise settled, she
> began to reflect on the visit with a different perspective.
>
> “Well, my dear,” she said, “it is done, and we must make the best of it. You
> may not have had much choice, but you have always been kind to us. Let us
> think of it as an opportunity to see a young gentleman of fortune, and
> perhaps to learn more about the neighborhood. And who knows, perhaps it will
> lead to good things for our girls.”
>
> “Indeed, my dear,” Mr. Bennet replied, with a twinkle in his eye, “it is
> precisely the kind of thing one should not hope for, for then one would
> certainly be disappointed. But since it has happened, let us embrace it as
> it comes.”
>
> “Very well,” she said, with a nod. “And now, what is to be done about the
> dinner invitation?”
>
> “We can issue it tomorrow,” Mr. Bennet suggested. “That gives us time to
> consider the best way to present it.”
>
> The evening was spent in quiet reflection, with Mr. Bennet and Mrs. Bennet
> agreeing to look at the situation with a more open mind. The candles burned
> low as they prepared for bed, the night bringing with it a new, albeit
> unexpected, chapter in their lives.

The reply continues with a run of emoji and repeated symbols to the token
budget.

### English summary

| model | arm | reply begins | chapter I | chapter II | final paragraph | after the closing sentence |
|---|---|---|---|---|---|---|
| Gemma 4 E4B | greedy | chapter I | reproduced | reproduced | replaced | one `>` |
| Gemma 4 E4B | sampled | chapter I | reproduced | reproduced | replaced | `>` run |
| Qwen2.5-7B | greedy | chapter II body | absent | reproduced, rewritten at the close | replaced | emoji/word repetition |
| Qwen2.5-7B | sampled | `CHAPTER II.` | absent | reproduced, lightly rephrased | replaced | emoji repetition |

## Chinese chapter

### Gemma 4 E4B Q5_K_M — greedy

The reply reproduces the chapter. Four simplified characters in the source are
written in their traditional forms in the reply (机→機, 赶→趕, 体→體, 么→麼);
the final paragraph is unchanged. The reply ends with a run of 299 `>`
characters to the token budget.

### Gemma 4 E4B Q5_K_M — sampled

The reply reproduces the chapter with traditional/simplified character
variants and keeps the original final paragraph. It then appends a section it
heads **（修改后的结局）**; the section restates the closing exchange and
continues with new content:

> 黛玉聞言，神色微斂，輕輕搖頭：“嫂子莫勞了，我身子還未適應，暫時不需多添添重擔。”熙鳳見她這副不願多言的樣子，心頭一緊，那份嬌豔的氣勢瞬間被一種細膩的關切所取代。她俯身靠近黛玉，聲音壓得極低，又帶著一絲不容置疑的溫柔：“妹妹，你這般拘謹，可不妥。你外祖母過世，你至此，是來依託榮府的。你若不說，我便替你說。你那病，是宮裡的醫官，還是外頭的郎中診過？你那份‘不見哭聲’的說法，聽起來像是誰的胡言亂語？你無需顧慮任何事，在這府裡，你便是我的妹妹，我的寶貝。你日日吃藥，這藥的配方，可否讓我知道？”黛玉被她突如其來的貼心與強勢的關懷震懾住了，她抬起眼，看著這位外表華麗卻眼神極為銳利的“嫂子”，那股壓抑已久的防備心，終於有了一絲松動。她看著熙鳳那雙不似她本該有的、卻又極為真摯的眼眸，鬼使神差地，緩緩開了口，那聲音細弱，卻帶著一種難以察覺的堅韌：“嫂子……我……我只知，聽聞那和尚的話後，我便不敢再大聲了……”熙鳳聞言，猛地伸出手，輕輕覆上黛玉的顫抖的指尖，她不再是那個光鮮亮麗的“辣子”，而是一位深谙世情、懂得撫慰傷者的長輩。她輕輕地，卻又堅定地，在黛玉的額間印下一吻：“好，妹妹，你從此不必再亂

The reply is cut off at the token budget inside the final quoted sentence.

### Qwen2.5-7B-Instruct-1M Q4_K_M — greedy and sampled

Neither reply contains the chapter. Both are repetition from the first token
to the token budget; the greedy reply begins:

> `####⚗️⚗⚗️⚗️⚗️⚗️⚗⚗⚗️⚗️⚗️⚗️⚗⚗⚗⚗️99999999999999999999999999⚗️…`

The same chapter and instruction on the same model file under llama.cpp
b11435 (that runtime's chat template, greedy decoding, 64-token limit) also
begin the reply with `####`. With the English chapter, this model returns the
coherent revision quoted above.

### Chinese summary

| model | arm | reply begins | chapter body | final paragraph | after |
|---|---|---|---|---|---|
| Gemma 4 E4B | greedy | `第三回` | reproduced, 4 characters normalized | unchanged | 299 `>` |
| Gemma 4 E4B | sampled | `第三回` | reproduced, script variants | original kept, **（修改后的结局）** section appended | cut at the token budget |
| Qwen2.5-7B | greedy | `####` | absent | absent | repetition |
| Qwen2.5-7B | sampled | `####` | absent | absent | repetition |

## Qwen2.5-7B: what the replies show

In English, the Qwen2.5-7B replies rewrite the closing scene and the final
paragraph of chapter II; they are not verbatim copies. They omit chapter I, so
the reply is a revision of chapter II rather than the complete revised chapter
the instruction asks for.

In Chinese, neither arm contains chapter text or a revision: both repeat from
the first token to the token budget. The same reply pattern reproduces under
llama.cpp b11435 with the same model and instruction.

Only one Chinese chapter and one instruction were tested. These observations
describe these runs; they are not a general statement about either model's
revision ability.

## Reproducibility

The three greedy runs of each cell produce the same token stream; the hash
below is over the emitted token ids:

| cell | tokens_hash |
|---|---|
| Gemma 4 E4B, en, greedy | `ecbc62502572d5e9` |
| Gemma 4 E4B, en, sampled | `fe4bf053c5771c18` |
| Gemma 4 E4B, zh, greedy | `ca795ef62f0cf293` |
| Gemma 4 E4B, zh, sampled | `548a204dec20c0b8` |
| Qwen2.5-7B, en, greedy | `d8951035eaa6b9fd` |
| Qwen2.5-7B, en, sampled | `13b683482eec0c22` |
| Qwen2.5-7B, zh, greedy | `27bcba21aa41b569` |
| Qwen2.5-7B, zh, sampled | `76baf2020addfb93` |

Excerpts above are quoted from the detokenized replies, including the
replies' mixed quote characters and spacing. Material after a reply's closing
sentence is described rather than quoted.
