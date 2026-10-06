# Kanoon-Bridge: demo video script (7 min 30 s)

**Speakers:** Gaurav (G), Sharanya (S), Kashvi (K), Shaurya (Sh).
**Setup before recording:**
- Run `make web` and open http://localhost:8000.
- Open one terminal in the project folder, font size 16 or more.
- Open the VS Code tabs listed in each section.
- Clear the chat history in the app and pick **Adv. Meera** as the advocate.
- Keep `results/figures/` open in an image viewer.

Timings add up to 7:30, so you have 30 s of slack. The words in quotes are what to say; the lines under **Screen** are what to show.

---

## 0:00–0:55 · The problem and the track (G)

**Screen:** the app's landing page with the bridge art. At 0:30 switch to a slide or the report's first page with the 302 example.

> "On 1 July 2024, India replaced the Indian Penal Code with the Bharatiya Nyaya Sanhita. Every section was renumbered. Murder was IPC 302; it is now BNS 103. And BNS 302 is a different offence: hurting religious feelings.
>
> So a search engine that matches the string '302' now returns the wrong law for every incident after July 2024. Precedents, meanwhile, are still written in IPC numbers, so a BNS-era query misses them all.
>
> Most people also describe their problem in Hindi or Hinglish, not in section numbers. And a judgment only binds you if it comes from the Supreme Court or your own High Court.
>
> Kanoon-Bridge is a vertical legal search engine built for exactly this: it is version-aware, multilingual and jurisdiction-aware. That's our track: T6, vertical search, in the legal domain, with a T5 layer for Hinglish."

---

## 0:55–2:35 · End to end on real queries, with a limitation (G drives, others watch)

**Screen:** the app.

1. Type `mere bhai ko chaku maara, kya case banta hai`, set **Delhi** and **01-03-2025**, and press Search.

   > "A Hinglish question. The chips show what the system understood: Hinglish, BNS in force, Delhi's binding courts. The top law is BNS 118, hurt by dangerous weapons. Notice it never said 'section' or 'BNS'.
   >
   > The advocate's answer quotes the sources, and every sentence is checked against the source it cites. The judgments are marked 'Binds Delhi' or 'Persuasive only'."

2. Click **How this was found**. Scroll slowly through the steps.

   > "This panel shows every step the backend took and its time in milliseconds: language detection, the Hindi lexicon mapping chaku to knife and stab, the code-in-force decision, BM25F, the statute bridge, authority. The whole thing takes about 100 ms."

3. Type `punishment under section 302` (2025 date) and press Search.

   > "The bare number '302'. The chips show both readings across the bridge. For a 2025 incident, BNS 302 is in force, so that comes first, and the answer flags 'Check the code' wherever a judgment cites old IPC 302."

4. Change the date to **01-03-2023** and search again.

   > "Same words, 2023 date: now IPC 302, murder, is the answer."

5. Type `punishmnt for cheeting` and press Search.

   > "Typos get a did-you-mean and the corrected search."

6. **The limitation.** Type `GST rate on restaurant food` and press Search.

   > "Something outside our corpus. Our corpus has no GST law, so the advocate refuses to write an answer instead of making one up. That's the abstention check."

   Then go back to the Hinglish result.

   > "One real limitation: about 12% of IL-PCSR judgments have no case title, so some show as 'Untitled judgment' with an ID. And in Hinglish, 'maara' can mean hit or kill, so murder sections still rank second. We have no dense model in this run to separate them."

---

## 2:35–3:40 · Indexing and query processing (S)

**Screen:** the terminal, then VS Code with `src/kanoon_bridge/index/positional.py` and `query/parser.py`.

```bash
python scripts/show_postings.py "IPC 302 murder knife"
```

> "This is the inverted index itself. Each term has its document frequency, idf, and postings with term frequency and positions.
>
> The key line is this one: 'sec:ipc:302' and 'off:murder_bns103' have the same postings. At index time, every section number is also indexed as an offence id shared by both codes. So a BNS 103 query and an IPC 302 judgment meet on the same posting list.
>
> At the bottom is the BM25F score for the top document, term by term: each zone's term frequency, length-normalised, weighted, then saturated with k1. These are the real numbers the ranker uses."

Show `positional.py` for 10 s: positions are stored as `array('I')`, zone indexes keep only counts, and memory dropped from 399 MB to 139 MB.

```bash
python app/cli.py '"dowry death" AND bail NOT anticipatory' --date 2023-05-01 -k 3 --no-snippets
```

> "The query parser is recursive descent. It handles Boolean operators, phrases, proximity like `murder /5 knife`, and facet filters like `state:delhi after:2015`. Postings are intersected rarest-first. One detail: a section term in a Boolean query matches its offence in either code."

---

## 3:40–4:45 · Hindi and Hinglish (K)

**Screen:** the app with `हत्या की सजा`, then `hatya ki saza kya hai`, then the terminal.

> "The same need typed three ways (English, Devanagari, Roman Hinglish) should find the same law. Here, both Devanagari and Hinglish find the murder sections."

Open **How this was found** on the Hinglish query.

> "The steps:
> - detect the language;
> - transliterate Devanagari to Roman with our own table;
> - normalise spellings, so 'maara', 'mara' and 'marra' match;
> - a Hindi soundex for aspirated consonants;
> - a legal lexicon of about 165 entries, like dahej → dowry and chaku → knife;
> - and a light Hindi stemmer."

Show `data/lexicons/hinglish_legal_terms.csv` and `text/transliterate.py` for 10 s each.

> "Our E4 set has 20 legal needs, each written in all three languages: 60 queries. The language ablation adds one step at a time. With no Hindi processing, Hindi and Hinglish queries score almost zero: P@5 of 0.00 and 0.01. Once the legal lexicon is on, Hindi reaches 0.23 and Hinglish 0.21, level with English at 0.21. The dense multilingual channel (NLLB-E5) is built but needs a GPU to encode, so it is off in these numbers."

---

## 4:45–5:45 · The research agent (Sh)

**Screen:** the app with the research-agent toggle ON. Ask `bail in dowry death case`, Maharashtra, 2023. Open **How this was found**.

```bash
python app/cli.py "bail in dowry death case" --agent --state maharashtra --date 2023-05-01 --debug -k 5 --no-snippets
```

> "Layer 2 plans several sub-queries:
> - the original question;
> - a cross-code rewrite, with IPC sections turned into BNS and back;
> - a Boolean query built from the top-idf terms;
> - one restricted to the courts that bind Maharashtra;
> - a statutes-first query.
>
> Each sub-query is an ordinary Layer 1 search, so the agent never ranks on its own. Results are merged with reciprocal rank fusion. Then it reflects. If query-performance prediction says retrieval is weak (low idf, a flat score head), it runs a pseudo-relevance-feedback round with Rocchio-style expansion terms.
>
> The 'from' column in the debug output shows which sub-query found each case."

---

## 5:45–7:00 · Ranking, answers and evaluation (G)

**Screen:** `docs/report/figures/` and `results/tables/main_results.csv`, then the report's results table.

> "Ranking is BM25F over the judgment zones, plus:
> - a statute bridge, which boosts precedents that applied the top-ranked statutes, in either code;
> - jurisdiction-aware PageRank, binding versus persuasive;
> - a learning-to-rank layer, coordinate ascent on MAP, trained only on the validation split.
>
> When the query is a whole pasted judgment, we add a word-trigram BM25 channel. Shared phrasing is the strongest signal there. We tuned BM25's b, the zone weights and the trigram weight on validation only, and never touched the test split."

Show `docs/report/figures/fig_main.png`.

> "E1 is IL-PCSR's own precedent-retrieval benchmark: 627 test queries, each a whole judgment. Plain BM25 gets MAP 0.22. We get 0.42, and F1 goes from 0.15 to 0.31. That is close to the best lexical baseline in the IL-PCSR paper, which is BM25 over word trigrams. Every gain is significant under a paired randomization test with Holm correction."

Show `fig_e3.png`.

> "The version-aware sets are where the design shows:
> - E3 is our core result. 'Cases under section 103 BNS': BM25 gets MAP 0.01, because no judgment says 'BNS 103'. We get 0.67.
> - E7 asks the same offence before and after July 2024. The right code at rank 1 goes from 47% to 94%.
> - E2 is colliding numbers like 302. Wrong-offence hits in the top 10 drop from 8% to zero."

Show `fig_ablation.png`.

> "The ablation adds one component at a time. Trigrams add 0.19 MAP. Learning to rank adds 0.03. And the statute bridge actually costs 0.03 on whole-judgment queries. We report that too."

Show the RAG line of `results/tables/rag.csv`.

> "For answers, 91% of sentences are supported by the source they cite. Abstention is still weak: it caught one of four out-of-scope questions. That's on our list."

---

## 7:00–7:30 · Close (all four on screen, or G)

> "To sum up: Kanoon-Bridge bridges the IPC-to-BNS change, reads Hinglish, and knows which courts bind you. Every result explains why it was retrieved. Next, we want to:
> - encode the dense channel on a GPU;
> - grow our judged sets with two judges and Cohen's kappa;
> - add CrPC → BNSS and the Evidence Act → BSA.
>
> Thank you."

---

### Checklist against the brief

| Requirement | Where in the video |
| --- | --- |
| Problem and track relevance, ≤ 1 min | 0:00–0:55 |
| End to end on real queries, with at least one limitation | 0:55–2:35 (Hinglish, 302 collision, date flip, typo, abstention, untitled judgments, "maara" ambiguity) |
| Pipeline with code and intermediate output (postings, weights, scores) | 2:35–3:40 `show_postings.py`; 4:45 agent `--debug`; "How this was found" panel |
| Evaluation against a baseline | 5:45–7:00 E1, E2, E3, E4, E7, ablation, significance |
| Each member explains their component | S 2:35, K 3:40, Sh 4:45, G 0:00 and 5:45 |
