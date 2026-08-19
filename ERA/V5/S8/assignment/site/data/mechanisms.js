/* ===========================================================
   THE SINGLE SOURCE OF TRUTH.

   Every date, source and claim the app makes comes from this file:
   the timeline (Act 2), the dossiers (Act 3), the sources table
   (Act 5) and the README all read from here. That is deliberate —
   four hand-maintained copies of twenty-one dates is exactly how a
   verified date silently drifts into a wrong one.

   Signed off by the user on 2026-08-19 after a row-by-row date audit
   (.lavish/s8-date-audit.html). Full evidence for every row lives in
   resources/chronology-research.md.

   Every arXiv date is the **v1 submission** timestamp, never the
   latest revision — a v3 posted in 2024 on a paper first submitted in
   2021 would silently reorder the whole timeline.

   Fields
     n        display number ('13b' for the split NTK row)
     id       anchor / deep-link slug
     date     ISO, exact
     lane     pos | mem | cmp | sys | origin   (which bill it pays)
     src      arXiv id, or 'reddit' / 'github' for non-paper sources
     url      canonical link
     paper    title of the primary source
     who      authors, short form
     one      one-line summary, used on the timeline hover
     idea     what it actually does
     buys / costs / pick     the honest-tradeoffs block
     v1       exact arXiv v1 submission timestamp, straight from the API
              (absent on the two rows that have no paper)
     tier     evidence class, carried over from the signed-off audit:
              A direct metadata · B citation chain · C archival
     verified how this date was checked  (shown on EVERY dossier)
     note     optional lineage / priority caveat
   =========================================================== */
window.MECHANISMS = [

{n:1, id:'learned-abs', date:'2016-11-07', lane:'pos', src:'1611.02344',
 url:'https://arxiv.org/abs/1611.02344',
 paper:'A Convolutional Encoder Model for Neural Machine Translation',
 who:'Gehring, Auli, Grangier, Dauphin',
 name:'Learned absolute positions',
 one:'A trainable lookup table with one vector per position, added to the token embedding.',
 idea:'Attention on its own is a bag of vectors — permute the tokens and the output is identical. The first fix is the most literal one available: keep a trainable table with one row per position index, look up row <code>i</code> for the token at position <code>i</code>, and add it to that token\'s embedding. The model learns whatever notion of position the data rewards.',
 buys:'Dead simple, and fully learned — no assumption is imposed about what position <em>means</em>, the data decides.',
 costs:'The table has a fixed number of rows. Position 4,097 in a 4,096-trained model does not have an embedding — it does not exist. There is no principled way to extrapolate a learned lookup, and it spends parameters on positions rather than on content.',
 pick:'Fixed, known, short context. This is exactly the wall Session 7 ran into, and the reason everything below exists.',
 v1:'2016-11-07T23:46:45Z',
 tier:'B',
 verified:'arXiv v1 metadata (2016-11-07T23:46:45Z). <strong>This date was corrected during the audit.</strong> The mechanism is usually credited to ConvS2S (2017-05-08) because that is what <em>Attention Is All You Need</em> cites — but AIAYN cites the later of two papers by the same group. This one states it outright: <em>"we add position embeddings to encode the absolute position of each source word within a sentence"</em>, form <code>e_j = w_j + l_j</code>.',
 note:'Antecedent, stated rather than buried: this paper credits Sukhbaatar et al. 2015 (End-To-End Memory Networks). That paper\'s "position encoding" is <em>fixed</em>, but its separate "Temporal Encoding" matrix is explicitly <em>"learned during training"</em> — a learned position embedding, except it indexes <strong>memory slots, not token positions</strong>. Different granularity, so the marker sits here.'},

{n:2, id:'sdpa', date:'2017-06-12', lane:'origin', src:'1706.03762',
 url:'https://arxiv.org/abs/1706.03762',
 paper:'Attention Is All You Need', who:'Vaswani et al.',
 name:'Scaled dot-product attention',
 one:'The baseline: score, scale, mask, softmax, weighted sum.',
 idea:'Every token emits three vectors — a <strong>query</strong> (what am I looking for), a <strong>key</strong> (what do I offer), a <strong>value</strong> (what I pass on). Score each query against every key with a dot product, divide by <code>√d_k</code> to keep the numbers out of softmax\'s saturated region, add <code>−∞</code> to future positions, softmax into weights, take the weighted sum of values. The <code>−∞</code> matters more than it looks: <code>softmax(−∞)=0</code> exactly is what lets every position train in parallel while staying causally honest.',
 buys:'Exact, content-based routing with a single hop between any two tokens — no distance penalty, no information bottleneck, and fully parallel at training time.',
 costs:'Both bills at once. <code>T²</code> score comparisons, and a KV cache that grows linearly in <code>T</code> for every concurrent user.',
 pick:'Always the starting point, and still the quality bar every mechanism below is measured against.',
 v1:'2017-06-12T17:57:34Z',
 tier:'A',
 verified:'arXiv v1 metadata. The baseline the whole app is built around; nothing in dispute.'},

{n:3, id:'sinusoidal', date:'2017-06-12', lane:'pos', src:'1706.03762',
 url:'https://arxiv.org/abs/1706.03762',
 paper:'Attention Is All You Need', who:'Vaswani et al.',
 name:'Sinusoidal encoding',
 one:'Fixed sin/cos waves at geometrically spaced frequencies instead of a learned table.',
 idea:'Same paper as the baseline, offered as the alternative to the learned table. Each position gets a fixed vector built from sine and cosine at a range of frequencies — fast-oscillating dimensions distinguish neighbours, slow ones separate distant regions. Nothing is trained.',
 buys:'Zero parameters, and — the important part — the function is <em>defined at every integer</em>, so nothing structurally breaks past training length. Offsets are also expressible as a linear transform of the encoding, which puts relative position within reach.',
 costs:'Defined is not competent. The model still never saw those positions during training, and quality degrades when you push past what it trained on. It also acts on the input embedding rather than on the query–key interaction itself.',
 pick:'A zero-parameter baseline. Mostly of historical interest now — RoPE does the relative-position job properly.',
 v1:'2017-06-12T17:57:34Z',
 tier:'A',
 verified:'Same paper and same submission as #2 — deliberately listed as its own entry, since the app treats them as two mechanisms that happen to share an origin.'},

{n:4, id:'sparse', date:'2019-04-23', lane:'cmp', src:'1904.10509',
 url:'https://arxiv.org/abs/1904.10509',
 paper:'Generating Long Sequences with Sparse Transformers',
 who:'Child, Gray, Radford, Sutskever',
 name:'Sparse / top-k attention',
 one:'Keep softmax, but let each query read a subset of keys instead of all of them.',
 idea:'The first serious attack on the compute bill that keeps attention itself intact. Rather than every query reading every key, restrict each query to a structured subset — strided and factorized patterns that still compose to full coverage across a couple of layers — or to the <code>top-k</code> highest-scoring keys.',
 buys:'Cuts the <code>T²</code> term toward roughly <code>T√T</code> while keeping softmax\'s sharp, selective retrieval. Sparsity is also empirically justified: real attention maps are already close to sparse.',
 costs:'A fixed pattern is a <em>guess</em> about where information lives, made before seeing the data. And genuine top-k has a chicken-and-egg problem: to know which scores are highest you must compute all the scores, so naive top-k saves memory traffic but not the scoring cost. You need a cheap proposal step to actually win — which #19 supplies, six years later.',
 pick:'When context is long and the access pattern really is local or structured. As a bolt-on at inference it disappoints; trained in, it works.',
 v1:'2019-04-23T19:29:47Z',
 tier:'A',
 verified:'arXiv v1 metadata.',
 note:'Priority, from the paper\'s own §2 Related Work — it names precursors rather than claiming to be first: <em>"(Parmar et al., 2018) uses blocks of local attention to apply Transformers to images"</em> (Image Transformer, 2018-02-15, ~14 months earlier), plus Transformer-XL for state reuse. Its own claim is breadth, not primacy: <em>"Our work is simpler than many of the techniques above and can be applied equally across images, text, and audio."</em> Credited here for <strong>factorized sparse attention at scale</strong>.'},

{n:5, id:'mqa', date:'2019-11-06', lane:'mem', src:'1911.02150',
 url:'https://arxiv.org/abs/1911.02150',
 paper:'Fast Transformer Decoding: One Write-Head is All You Need', who:'Shazeer',
 name:'MQA — Multi-Query Attention',
 one:'All query heads share one single K/V head.',
 idea:'The first attack on the memory bill, and a blunt one: keep all the query heads, but give the whole layer a <em>single</em> key head and a single value head that every query head reads from. The <code>kv_heads</code> term in the cache formula becomes 1.',
 buys:'Enormous cache reduction — at the lesson\'s yardstick, 6.44 GB collapses to <strong>805 MB</strong>, an 8× cut. Decoding is memory-bandwidth-bound, so this translates directly into faster generation.',
 costs:'Quality drops. Heads lose their independent views of the sequence — they can still ask different questions, but they all read the same answer sheet. Training instability was also reported.',
 pick:'When decode memory dominates everything and you can absorb the quality hit. Mostly superseded by GQA, which found the middle.',
 v1:'2019-11-06T00:19:05Z',
 tier:'A',
 verified:'arXiv v1 metadata. Single-author Google paper — and the timeline\'s best surprise, landing ~3.5 years before GQA.'},

{n:6, id:'swa', date:'2020-04-10', lane:'cmp', src:'2004.05150',
 url:'https://arxiv.org/abs/2004.05150',
 paper:'Longformer: The Long-Document Transformer', who:'Beltagy, Peters, Cohan',
 name:'Sliding window attention',
 one:'Each token attends only to the w tokens immediately before it.',
 idea:'The simplest sparsity pattern, with the cleanest justification: language is mostly local. Fix a window <code>w</code> and let each token see only that far back. Information still travels further because layers stack — <code>L</code> layers give an effective receptive field around <code>L × w</code>, the way a CNN grows its field with depth.',
 buys:'Compute becomes linear in <code>T</code>, and — the underrated part — the KV cache becomes <strong>bounded</strong> at <code>w</code> rather than growing. One of very few entries that flattens the memory bill rather than just lowering its slope.',
 costs:'Anything outside the window is reachable only indirectly, through layers, which is lossy. Exact long-range lookup — "what was the name in paragraph one" — is gone.',
 pick:'When local structure dominates, or paired with a global path for the few tokens that need reach.',
 v1:'2020-04-10T17:54:09Z',
 tier:'A',
 verified:'arXiv v1 metadata.',
 note:'Longformer does not claim the pattern. Its own words: <em>"The model with the most similar attention pattern to ours is Sparse Transformer (Child et al., 2019), which uses a form of dilated sliding window"</em>. Its stated novelty is a more flexible CUDA kernel plus <em>"task motivated <strong>global attention</strong> patterns… essential for good performance in the transfer learning setting"</em>. So the honest credit is sliding window <strong>+ global tokens</strong> as a named NLP mechanism. Mistral 7B (2023-10-10) is the famous deployment, not the origin.'},

{n:7, id:'linear', date:'2020-06-29', lane:'cmp', src:'2006.16236',
 url:'https://arxiv.org/abs/2006.16236',
 paper:'Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention',
 who:'Katharopoulos, Vyas, Pappas, Fleuret',
 name:'Linear attention',
 one:'Remove softmax so the matrix products reassociate into a fixed-size recurrent state.',
 idea:'Softmax is what forces the <code>T×T</code> matrix to exist: it couples every score in a row through the normaliser, so you cannot factor it away. Drop it, replace it with a kernel feature map <code>φ</code>, and associativity becomes available — <code>(φ(Q)φ(K)ᵀ)V</code> can be computed as <code>φ(Q)(φ(K)ᵀV)</code>. The bracketed term is a fixed <code>d×d</code> state that does not depend on <code>T</code> at all.',
 buys:'<code>O(T)</code> compute, and a <strong>constant-size</strong> state instead of a growing cache. The memory bill stops growing entirely — not a lower slope, a flat line. Generation becomes RNN-like: one fixed state, updated per token.',
 costs:'Softmax was doing real work. Its sharp, near-winner-take-all selection is what makes exact retrieval possible; a smooth kernel blurs it. Worse, a state that only <em>accumulates</em> writes new information on top of old and smears the two, so recall degrades as the sequence grows.',
 pick:'When throughput and memory dominate and perfect recall does not — and in practice, hybridised with real attention layers rather than used alone.',
 v1:'2020-06-29T17:55:38Z',
 tier:'A',
 verified:'arXiv v1 metadata.',
 note:'Priority — the audit\'s closest call. Shen et al., <em>"Efficient Attention: Attention with Linear Complexities"</em> (arXiv 1812.01243, v1 <strong>2018-12-04</strong>) already reassociates the product for linear complexity, <strong>19 months earlier</strong>; Katharopoulos et al. cite it and call it <em>"concurrent"</em>, which holds for venues but not for arXiv v1 dates. The marker stays here because the lesson\'s argument needs the <em>causal</em> fixed-state recurrence — and Efficient Attention is a vision paper that never builds it. A full-text scan of that PDF finds exactly <strong>one</strong> occurrence of "causal"/"autoregressive", inside a bibliography entry.'},

{n:8, id:'delta', date:'2021-02-22', lane:'cmp', src:'2102.11174',
 url:'https://arxiv.org/abs/2102.11174',
 paper:'Linear Transformers Are Secretly Fast Weight Programmers',
 who:'Schlag, Irie, Schmidhuber',
 name:'Delta rule (origin)',
 one:'Before writing to the fixed state, read what it already says and write only the difference.',
 idea:'The fix for linear attention\'s smearing problem, borrowed from associative memory. Instead of blindly adding the new key–value pair into the state, first <em>query the state with the incoming key</em> to see what it currently associates with it, then write only the correction — the delta between what you wanted stored and what is already there. Updating a key now <strong>replaces</strong> its old association rather than stacking on top of it.',
 buys:'Turns a fixed-size state from an accumulator into usable memory — the difference between a state that degrades with every write and one that can be maintained.',
 costs:'Read-then-write is inherently sequential, which is what a GPU hates — so it stayed impractical to train at scale for three years, until #17. And capacity is still fixed: correction manages the budget well, it does not enlarge it.',
 pick:'Any time you run a fixed-size recurrent state and need it to survive long sequences. In modern practice you would use the gated version (#18).',
 v1:'2021-02-22T16:51:38Z',
 tier:'A',
 verified:'arXiv v1 metadata.',
 note:'<strong>Disputed priority, kept visible on the timeline as an arc.</strong> This is the actual origin of the delta-rule correction — it predates the paper everyone calls "the DeltaNet paper" (#17) by over three years, and even predates RoPE.'},

{n:9, id:'rope', date:'2021-04-20', lane:'pos', src:'2104.09864',
 url:'https://arxiv.org/abs/2104.09864',
 paper:'RoFormer: Enhanced Transformer with Rotary Position Embedding', who:'Su et al.',
 name:'RoPE',
 one:'Rotate queries and keys by an angle proportional to position; the dot product then depends only on i−j.',
 idea:'The insight that made everything else about position tractable. Treat each pair of dimensions as a 2D plane and rotate the query and key vectors by an angle proportional to their position. Because rotating two vectors and taking their dot product leaves only the <em>difference</em> of the angles, the score between positions <code>i</code> and <code>j</code> becomes a function of <code>i−j</code> alone. Relative position falls out of the geometry rather than being added on.',
 buys:'Relative position for free, inside the score itself. No table, no extra parameters, no added input vector, and mathematically defined at any position. It is the default in essentially every modern open model.',
 costs:'Rotation frequencies are fixed at training time. Push past the trained length and the low-frequency dimensions produce angles the model has never seen — defined everywhere, competent only where it trained. That single limitation generates #13, #13b, #14 and arguably #20.',
 pick:'Default for any new decoder. The question is not whether to use RoPE, it is what you do about extension.',
 v1:'2021-04-20T09:54:06Z',
 tier:'A',
 verified:'arXiv v1 metadata.'},

{n:10, id:'alibi', date:'2021-08-27', lane:'pos', src:'2108.12409',
 url:'https://arxiv.org/abs/2108.12409',
 paper:'Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation',
 who:'Press, Smith, Lewis',
 name:'ALiBi',
 one:'Skip positional embeddings entirely; subtract a penalty proportional to distance from each score.',
 idea:'The contrarian answer, four months after RoPE. Add no positional information to the vectors at all. Instead, after scoring, subtract a penalty proportional to the distance <code>i−j</code>, with a different fixed slope per head — so some heads look sharply local and others tolerate distance.',
 buys:'Startlingly simple: no embeddings, no parameters, a few lines of code. And it extrapolates — train at 1K, run at 2K+, with graceful rather than catastrophic degradation, which was the paper\'s whole point.',
 costs:'It bakes in a <strong>recency prior as a structural fact</strong>. A distant token is penalised for being distant regardless of relevance, and the model cannot learn to override it. Fine for perplexity, bad for long-range retrieval — which is precisely the workload long context is supposed to serve.',
 pick:'When cheap length extrapolation matters more than distant recall.',
 v1:'2021-08-27T17:35:06Z',
 tier:'A',
 verified:'arXiv v1 metadata.'},

{n:11, id:'flash', date:'2022-05-27', lane:'sys', src:'2205.14135', bonus:true,
 url:'https://arxiv.org/abs/2205.14135',
 paper:'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
 who:'Dao, Fu, Ermon, Rudra, Ré',
 name:'FlashAttention',
 one:'Identical math, rescheduled so the T×T matrix never touches slow memory.',
 idea:'The bonus mechanism, and the odd one out. Every other entry changes <em>what</em> attention computes; this changes only <em>where the numbers live while it computes them</em>. Standard implementations materialise the full <code>T×T</code> matrix in slow HBM, write it, read it back for softmax, write again. FlashAttention tiles Q, K and V into blocks small enough for fast on-chip SRAM and fuses score → softmax → weighted-sum into one pass, using <strong>online softmax</strong> — a running max and running sum — so it never needs a whole row at once and never materialises the matrix at all.',
 buys:'Exact attention, bit-for-bit. <strong>Zero accuracy cost</strong> — the only entry here that gives something up for nothing. Large wall-clock speedup, and memory drops from <code>O(T²)</code> to <code>O(T)</code>. A drop-in kernel swap; the architecture does not change.',
 costs:'It does not reduce FLOPs — the compute bill\'s arithmetic is untouched, only the memory traffic around it. It does nothing for the inference KV cache, so the memory bill is exactly as bad as before. And it is hardware-specific: tile sizes are tuned to particular SRAM capacities, which is why there is a FlashAttention-2 and -3 rather than one portable idea.',
 pick:'Essentially always, on supported hardware. The interesting question is not when you would choose it but why you would ever not.',
 v1:'2022-05-27T17:53:09Z',
 tier:'A',
 verified:'arXiv v1 metadata, <strong>plus</strong> an independent <code>pdftotext</code> read of the PDF title page — checked twice because this is the bonus row and carries extra scrutiny.',
 note:'<strong>Bonus mechanism, not covered in class.</strong> Chosen because it answers a third cost the lesson never names — memory <em>bandwidth</em> — rather than either of the two bills, and because it lands in the 2021–2023 gap where the minimum-coverage list looks empty.'},

{n:12, id:'gqa', date:'2023-05-22', lane:'mem', src:'2305.13245',
 url:'https://arxiv.org/abs/2305.13245',
 paper:'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints',
 who:'Ainslie et al.',
 name:'GQA — Grouped-Query Attention',
 one:'Query heads are split into G groups; each group shares one K/V head.',
 idea:'The middle ground between full multi-head and MQA, arriving three and a half years after MQA. Pick a group count — 8 query heads per KV head is typical — and let each group share. MHA and MQA are the two endpoints of the same dial.',
 buys:'Most of MQA\'s savings at most of MHA\'s quality: at the yardstick, dropping 8 KV heads to 2 takes 6.44 GB to <strong>1.61 GB</strong>. It can also be <em>uptrained</em> from an existing MHA checkpoint for a small fraction of original training compute — you do not start over. That practicality is why it is everywhere.',
 costs:'It lowers the slope of the cache line; it does not flatten it. Double the context and you still double the cache. A constant factor, not a different asymptotic.',
 pick:'The default for production decoders today. If you are shipping a model and have not chosen otherwise, this is the answer.',
 v1:'2023-05-22T17:16:38Z',
 tier:'A',
 verified:'arXiv v1 metadata.'},

{n:13, id:'ntk', date:'2023-06-29', lane:'pos', src:'reddit',
 url:'https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/',
 paper:'"NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size…" — r/LocalLLaMA',
 who:'u/bloc97',
 name:'NTK-Aware scaling',
 one:'Extend a RoPE model by stretching low frequencies while leaving high ones nearly intact.',
 idea:'Naive position interpolation squeezes all positions into the trained range by scaling every RoPE frequency equally — which works, but crushes the high-frequency dimensions that distinguish adjacent tokens, so the model loses local precision. The NTK-aware fix scales the RoPE <em>base</em> instead, stretching slow dimensions a lot and fast ones almost not at all: long-range reach without destroying short-range detail.',
 buys:'Real context extension with <strong>no fine-tuning at all</strong> and a few lines of code. It went from a forum post to production practice in weeks, which tells you how badly the problem needed solving.',
 costs:'A heuristic, not a derivation — quality decays as you push the scale factor, with no principled stopping point. It arrived with no paper, no ablations and no independent evaluation.',
 pick:'When you need more context out of a model you already have, today, without a training run.',
 tier:'C',
 verified:'<strong>No paper exists — the weakest evidence class in the set, and still pinned to the minute.</strong> reddit.com is bot-walled to both <code>curl</code> and a real browser, so the date came from the Wayback Machine two independent ways: a snapshot carries Reddit\'s own <code>created-timestamp="2023-06-29T08:21:29.413+0000"</code> with <code>author="bloc97"</code>, and the CDX index\'s earliest capture of the URL is 08:21:50 UTC the same day — an archive cannot predate what it archived.',
 note:'A third community method also exists — <strong>"Dynamic NTK"</strong> by emozilla, a different author, which YaRN cites as the other improvement on NTK-Aware. It is deliberately not given its own entry: it is outside the assignment\'s coverage list, and its date could not be pinned to the standard every other entry here meets.'},

{n:'13b', id:'ntk-parts', date:'2023-07-07', lane:'pos', src:'github',
 url:'https://github.com/jquesnelle/yarn/pull/1',
 paper:'"Add NTK-Aware interpolation \'by parts\' correction" — PR #1 to jquesnelle/scaled-rope',
 who:'bloc97',
 name:'NTK-By-Parts',
 one:'Refines NTK-Aware by treating RoPE dimensions differently according to their wavelength.',
 idea:'The correction that made the NTK family actually work. Rather than one global base change, it splits the RoPE dimensions into groups by wavelength: dimensions whose period is short relative to the trained context are left <em>untouched</em>, dimensions whose period exceeds it are fully interpolated, and a band in between is blended. Local precision is preserved exactly where plain NTK-Aware was still degrading it.',
 buys:'Better quality than both position interpolation and plain NTK-Aware, with or without fine-tuning — and it is what YaRN is built on, so it sits directly on the main line of descent.',
 costs:'Introduces two tuning parameters that must be set per model family, and like everything in this group it remains a heuristic correction to a frequency scheme rather than a model trained long.',
 pick:'Over plain NTK-Aware, essentially always — it is strictly the better-studied of the two.',
 tier:'B',
 verified:'<strong>Not a paper and not a Reddit post — a GitHub pull request</strong>, which makes this the best-evidenced non-paper row here. Dated from GitHub\'s own API, two independent timestamps: the PR was opened by <code>bloc97</code> at <code>2023-07-07T20:40:33Z</code> and its first commit is authored <code>2023-07-07T20:24:12Z</code>, 16 minutes earlier the same day. Merged 2023-07-09. Platform record, not archival inference. The repo was later renamed, so YaRN\'s cited URL now resolves to <code>jquesnelle/yarn/pull/1</code>.',
 note:'Split out from #13 during the date audit. <strong>YaRN builds on this, not on plain NTK-Aware</strong> — 2309.00071 §3.2 says <em>"a variant of the resulting method was released under the name \'NTK-by-parts\' interpolation"</em>. The lineage arrow points here.'},

{n:14, id:'yarn', date:'2023-08-31', lane:'pos', src:'2309.00071',
 url:'https://arxiv.org/abs/2309.00071',
 paper:'YaRN: Efficient Context Window Extension of Large Language Models',
 who:'Peng, Quesnelle, Fan, Shippole',
 name:'YaRN',
 one:'The NTK family, formalised: interpolate by wavelength band, plus a temperature correction.',
 idea:'Takes the community\'s NTK heuristics and turns them into a method. The core move is to stop treating all RoPE dimensions the same: short-wavelength dimensions are left alone, dimensions whose wavelength exceeds the trained context are fully interpolated, and a band between is blended. On top of that it scales attention temperature, correcting a distribution shift that interpolation alone introduces.',
 buys:'The best quality-per-effort in the extension family. It works with no fine-tuning and better with a small one — the paper reports needing roughly 10× less data and fewer steps than earlier extension methods to reach a given quality.',
 costs:'Still extension of a model pretrained short. The weights never saw a genuine 100K-token dependency during training, so you are widening the aperture on a model never taught to use it. Exactly the "two roads" tension: extend a short-trained model, or train long natively.',
 pick:'The strongest option when you must extend an existing checkpoint rather than train a new one.',
 v1:'2023-08-31T18:18:07Z',
 tier:'A',
 verified:'arXiv v1 metadata. Its own bibliography is also what established that #13 and #13b have no papers, and what separated the two.'},

{n:15, id:'sinks', date:'2023-09-29', lane:'mem', src:'2309.17453',
 url:'https://arxiv.org/abs/2309.17453',
 paper:'Efficient Streaming Language Models with Attention Sinks (StreamingLLM)',
 who:'Xiao, Tian, Chen, Han, Lewis',
 name:'Attention sinks',
 one:'Keep the first few tokens pinned in the cache forever; evicting them is what breaks streaming.',
 idea:'Starts from a strange observation: models dump enormous attention mass onto the first few tokens regardless of what they say. The explanation is mechanical — softmax must sum to 1, so a head with nothing it wants to attend to still has to put its probability <em>somewhere</em>, and the earliest tokens are visible to every position. They become a no-op drain. The consequence: in a rolling-window cache, the moment token 0 is evicted, quality collapses — not because it mattered semantically, but because the drain is gone.',
 buys:'Pin four or so sink tokens plus a rolling window and you get <strong>effectively unbounded streaming on a bounded cache</strong>, with no fine-tuning and no architecture change. The failure it prevents is dramatic.',
 costs:'It enables streaming, not long-context <em>recall</em>. Evicted content is genuinely gone; the model stays fluent forever but cannot answer questions about what scrolled off. Easy to mistake one for the other.',
 pick:'Long-running conversational or streaming deployments where the recent window is what matters.',
 v1:'2023-09-29T17:59:56Z',
 tier:'A',
 verified:'arXiv v1 metadata.'},

{n:16, id:'mla', date:'2024-05-07', lane:'mem', src:'2405.04434',
 url:'https://arxiv.org/abs/2405.04434',
 paper:'DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model',
 who:'DeepSeek-AI',
 name:'MLA — Multi-head Latent Attention',
 one:'Cache one small low-rank latent per token; reconstruct K and V from it at use time.',
 idea:'Rather than sharing K/V heads, compress them. Project keys and values down into a single low-dimensional latent vector per token, cache <em>that</em>, and project back up to per-head K and V inside the attention computation. The up-projection matrices can be folded into neighbouring weights, so much of the reconstruction is free at inference.',
 buys:'A far larger cache reduction than GQA at <em>better</em> reported quality than full multi-head — the surprising part. It attacks the <code>kv_heads × head_dim</code> product directly instead of just the head count.',
 costs:'Considerably more complex, and it does not compose cleanly with RoPE — rotation does not survive the compression, so a decoupled rotary component must be carried alongside the latent. It also has to be designed in from pretraining; you cannot uptrain into it as easily as GQA.',
 pick:'When you control the architecture from scratch and serving cache is the binding constraint.',
 v1:'2024-05-07T15:56:43Z',
 tier:'A',
 verified:'arXiv v1 metadata. The mechanism is introduced inside a model paper rather than one of its own — normal for DeepSeek.'},

{n:17, id:'deltanet', date:'2024-06-10', lane:'cmp', src:'2406.06484',
 url:'https://arxiv.org/abs/2406.06484',
 paper:'Parallelizing Linear Transformers with the Delta Rule over Sequence Length',
 who:'Yang, Wang, Zhang, Shen, Kim',
 name:'DeltaNet (parallel training)',
 one:'Reformulates the delta rule\'s sequential update so it trains with a chunked parallel scan.',
 idea:'The delta rule (#8) was sound but sequential — each step needs the state the previous step produced, leaving the GPU idle. This paper rewrites the chain of rank-one corrective updates into a matrix-product form evaluable in chunks, so the sequence dimension parallelises. Same rule, tractable at scale.',
 buys:'Makes the 2021 idea actually trainable on modern hardware. This is what moved the delta rule from an interesting result to a viable architecture — and why "DeltaNet" is dated 2024 in most people\'s heads.',
 costs:'A chunked scan is still more machinery than a plain matmul, and the state capacity limit is untouched — you have made the update efficient, not the memory bigger.',
 pick:'Any fixed-state architecture you intend to actually train.',
 v1:'2024-06-10T17:24:42Z',
 tier:'A',
 verified:'arXiv v1 metadata.',
 note:'Commonly cited as the origin of the delta rule. It is not — see #8. The contribution here is the training algorithm, and the three-year gap between the two is one of the timeline\'s clearest findings.'},

{n:18, id:'gated-delta', date:'2024-12-09', lane:'cmp', src:'2412.06464',
 url:'https://arxiv.org/abs/2412.06464',
 paper:'Gated Delta Networks: Improving Mamba2 with Delta Rule',
 who:'Yang, Kautz, Hatamizadeh',
 name:'Gated DeltaNet',
 one:'Delta-rule correction plus a data-dependent forgetting gate — precise edits and bulk clearing.',
 idea:'Combines the two ways a fixed-size state can be managed, which turn out to be complementary rather than redundant. A <strong>gate</strong> decides how much of the whole state to decay — good for "the topic changed, clear the slate". The <strong>delta rule</strong> makes a targeted correction to one association — good for "this specific fact was updated". Gating alone is too blunt; delta alone cannot forget in bulk.',
 buys:'Meaningfully better recall and in-context retrieval than either Mamba2-style gating or plain DeltaNet alone, at the same constant memory.',
 costs:'Still a fixed-size state, so it still has a capacity ceiling real attention does not. In practice it is deployed <em>hybridised</em> — interleaved with genuine attention layers — which is an honest admission that the fixed state cannot do everything.',
 pick:'The current best-in-class fixed-state layer, used as the cheap layers in a hybrid depth schedule.',
 v1:'2024-12-09T13:09:04Z',
 tier:'A',
 verified:'arXiv v1 metadata. A "Gated DeltaNet-2" follow-up exists (2605.22791, 2026-05-21) — <strong>not</strong> the one the brief means; noted so the two are not swapped.'},

{n:19, id:'nsa', date:'2025-02-16', lane:'cmp', src:'2502.11089',
 url:'https://arxiv.org/abs/2502.11089',
 paper:'Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention',
 who:'Yuan, Gao, Dai, … Ruan, Liang (DeepSeek-AI)',
 name:'DeepSeek NSA',
 one:'Three gated branches — compressed blocks, top-k selected blocks, sliding window — trained in from scratch.',
 idea:'The answer to top-k attention\'s chicken-and-egg problem (#4). Each query reads through three parallel paths combined by a learned gate: a <strong>compressed</strong> branch summarising whole blocks into single representatives for a cheap global view; a <strong>selection</strong> branch using those cheap block-level scores to pick the top-k blocks worth reading at full fidelity; and a <strong>sliding window</strong> for local detail. The compressed branch is the cheap proposal step that makes selection affordable — you never score all keys at full resolution.',
 buys:'Sparsity that is <em>trained in</em> rather than bolted on, so the model learns to use it instead of being degraded by it. Kernels are written to match GPU memory access patterns, so theoretical savings show up as real measured speedups — the thing most sparse-attention papers cannot claim.',
 costs:'Substantially more machinery than anything above it, and selection is block-granular, so it is coarse — you retrieve a neighbourhood, not a token. It must also be present from pretraining; you cannot convert a dense checkpoint into it.',
 pick:'Training a long-context model from scratch where prefill and decode cost both matter.',
 v1:'2025-02-16T11:53:44Z',
 tier:'A',
 verified:'arXiv v1 metadata. Author list includes DeepSeek-AI leadership, confirming this is the paper behind the lesson\'s §12 compression + top-k block selection design.'},

{n:20, id:'drope', date:'2025-12-13', lane:'pos', src:'2512.12167',
 url:'https://arxiv.org/abs/2512.12167',
 paper:'Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings',
 who:'Gelberg, Eguchi, Akiba, Cetin (Sakana AI)',
 name:'DroPE',
 one:'Extend context by removing the positional embeddings rather than rescaling them.',
 idea:'The most recent answer, and it inverts the question. Everything from #13 through #14 tries to <em>rescale</em> RoPE so its frequencies survive past the trained length. DroPE removes the positional embedding instead, on the grounds that a causal decoder already carries order implicitly — a token can only see its past, so masking and depth encode sequence structure without an explicit signal. Take away the thing that breaks past its trained range, and there is nothing left to break.',
 buys:'Sidesteps the entire frequency-heuristic ladder. Reported at 8K → 256K, a 32× extension, on the LightningLM run.',
 costs:'That 32× needs reading carefully: it was applied <em>before</em> annealing, so it describes the range at which the model is <strong>defined</strong>, not the range at which it is demonstrably <strong>competent</strong>. Those are different claims and the lesson is emphatic about not conflating them. It is also very recent, with limited independent replication.',
 pick:'Worth watching rather than defaulting to. The interesting part is the direction of travel: after nine years of increasingly elaborate positional machinery, the newest idea is to delete it.',
 v1:'2025-12-13T04:23:47Z',
 tier:'B',
 verified:'arXiv v1 metadata — <strong>but this is the row I got wrong first.</strong> My initial search concluded DroPE had no public source, which was simply stopping too early. Found via LightningLM\'s bibliography, extracted two independent ways (a RAG hit <em>and</em> a raw <code>pdftotext</code> dump of the reference pages, bypassing chunking entirely), then title/authors/date re-confirmed against arXiv directly.'}

];

/* The V4 / LightningLM context marker — not one of the twenty-one mechanisms.
   Included because the app quotes its numbers, and those numbers should be
   traceable to a real published model rather than to course folklore. */
window.CONTEXT_MARKER = {
  id:'lightninglm', date:'2026-06-05', src:'2606.07404',
  url:'https://arxiv.org/abs/2606.07404',
  name:'LightningLM ("V4")',
  paper:'Reversible Foundations: Training a 120B Sparse MoE through State-Preserving Scaling',
  who:'Rohan Shravan, The School of AI',
  one:'The lesson\'s "V4" is a real published model, not a hypothetical.',
  v1:'2026-06-05T15:48:42Z',
  tier:'A',
  verified:'arXiv v1 metadata. <strong>This date was corrected during the Act 5 re-check</strong> — it had been recorded as 2026-06-08, three days late, from the listing page rather than the submission timestamp. Every V4-specific number the app quotes — the DDDGDDDG depth schedule, the KV-cache figures, the Memory Stream, the 8K→256K DroPE application — is grounded here.'
};

window.LANES = {
  pos:    {name:'Position &amp; length', short:'position', color:'var(--pos)'},
  mem:    {name:'KV-cache memory',      short:'memory',   color:'var(--mem)'},
  cmp:    {name:'Quadratic compute',    short:'compute',  color:'var(--cmp)'},
  sys:    {name:'Systems &amp; kernels', short:'systems',  color:'var(--sys)'},
  origin: {name:'The baseline',          short:'baseline', color:'var(--origin)'}
};
