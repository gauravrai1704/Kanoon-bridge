"""E22: how much of a legal RAG prompt is reused, and what a tiered KV cache would save.  [owner: Gaurav — working]

Every RAG answer pre-fills an LLM with retrieved chunks. Legal questions keep retrieving the
same few hundred statute sections, so the KV cache of a chunk computed for one question can be
reused by the next instead of being recomputed. This module measures that reuse on our own
traffic and simulates caches of a given size, as the workload study behind:

    PACT-KV (Rai, "Phase-Aware Cost-Optimal Tiering for LLM KV Cache"; github.com/gauravrai1704/
            tier-aware-kv-cache): GPU VRAM (T1) / CPU DRAM (T2) tiering of KV blocks
    RAGCache (Jin et al., 2024) and CacheBlend (Yao et al., EuroSys 2025): reusing the KV cache
            of retrieved documents across requests (CacheBlend: at any position, with partial
            recomputation)

Trace: the chunks (doc id + text hash) that RAG v2 puts in the prompt, question by question, for
a stream of questions (shuffled with a fixed seed). Chunk size in tokens ~ 1.3 x words.

Measured:
    reuse      share of prompt-chunk tokens already seen earlier in the stream (an infinite
               cache's hit rate), statute vs precedent chunks; share of retrievals taken by the
               most frequent 10% of chunks
    caches     hit rate (share of chunk tokens served from cache instead of recomputed) for a
               cache of C tokens under LRU, LFU, and STATUTE-PINNED (statute chunks kept in T1
               by frequency, precedents in an LRU T2), for C in KV-memory terms: a Mistral-7B KV
               entry is 128 KiB per token (32 layers x 8 KV heads x 128 dims x K,V x 2 bytes), so
               1 GiB holds ~8,192 tokens
Assumption: chunk KV is reusable at any position (CacheBlend-style); exact-prefix reuse only
would give less.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, OrderedDict

KV_BYTES_PER_TOKEN = 32 * 8 * 128 * 2 * 2          # Mistral-7B FP16 (PACT-KV's hardware track)


def tokens(text: str) -> int:
    return max(1, round(len((text or "").split()) * 1.3))


def collect(engine, rag, agent, questions, seed: int = 0, log=None) -> list[list[tuple[str, str, int]]]:
    """[[(chunk id, zone, tokens), ...] per question], questions in a shuffled order."""
    from kanoon_bridge.schema import Query
    from kanoon_bridge.search import SearchOptions

    qs = list(questions)
    random.Random(seed).shuffle(qs)
    trace = []
    for i, row in enumerate(qs):
        q = Query(row["text"], incident_date=row.get("incident_date"))
        ans = rag.answer(agent.run(q, SearchOptions.interactive()), generator="extractive", version=2)
        trace.append([(f"{c.doc_id}#{hashlib.md5(c.text.encode()).hexdigest()[:10]}",
                       "statute" if c.zone == "statute" else "precedent", tokens(c.text)) for c in ans.chunks])
        if log and (i + 1) % 100 == 0:
            log(f"  {i + 1}/{len(qs)}")
    return trace


def reuse(trace) -> dict:
    seen, hit, tot = set(), Counter(), Counter()
    freq = Counter()
    for req in trace:
        for cid, zone, n in req:
            tot[zone] += n
            freq[cid] += 1
            if cid in seen:
                hit[zone] += n
            seen.add(cid)
    top = sorted(freq.values(), reverse=True)
    k = max(1, len(top) // 10)
    out = {"requests": len(trace), "distinct_chunks": len(freq), "chunk_retrievals": sum(top),
           "top10pct_share": round(sum(top[:k]) / max(1, sum(top)), 4)}
    for z in ("statute", "precedent"):
        out[f"{z}_tokens"] = tot[z]
        out[f"{z}_reuse"] = round(hit[z] / tot[z], 4) if tot[z] else 0.0
    all_t = sum(tot.values())
    out["all_reuse"] = round(sum(hit.values()) / all_t, 4) if all_t else 0.0
    return out


def simulate(trace, capacity: int, policy: str) -> float:
    """Token hit rate of a chunk-KV cache of `capacity` tokens. policy: lru | lfu | pinned."""
    hits = total = 0
    if policy == "lru":
        cache, used = OrderedDict(), 0
        for req in trace:
            for cid, _, n in req:
                total += n
                if cid in cache:
                    hits += n
                    cache.move_to_end(cid)
                    continue
                cache[cid] = n
                used += n
                while used > capacity and cache:
                    _, m = cache.popitem(last=False)
                    used -= m
        return hits / total if total else 0.0
    if policy == "lfu":
        cache, freq, used = {}, Counter(), 0
        for req in trace:
            for cid, _, n in req:
                total += n
                freq[cid] += 1
                if cid in cache:
                    hits += n
                    continue
                cache[cid] = n
                used += n
                while used > capacity and len(cache) > 1:
                    victim = min((c for c in cache if c != cid), key=lambda c: freq[c])
                    used -= cache.pop(victim)
        return hits / total if total else 0.0
    if policy == "pinned":
        # T1: statute chunks, LFU (the law is the stable, shared part of every prompt);
        # T2: precedents, LRU; capacity split by the statute share of the traffic seen so far
        t1, t2, freq = {}, OrderedDict(), Counter()
        u1 = u2 = 0
        stat_tok = all_tok = 0
        for req in trace:
            for cid, zone, n in req:
                total += n
                all_tok += n
                freq[cid] += 1
                if zone == "statute":
                    stat_tok += n
                cap1 = capacity * max(0.5, stat_tok / all_tok)
                cap2 = capacity - cap1
                if cid in t1 or cid in t2:
                    hits += n
                    if cid in t2:
                        t2.move_to_end(cid)
                    continue
                if zone == "statute":
                    t1[cid] = n
                    u1 += n
                    while u1 > cap1 and len(t1) > 1:
                        victim = min((c for c in t1 if c != cid), key=lambda c: freq[c])
                        u1 -= t1.pop(victim)
                else:
                    t2[cid] = n
                    u2 += n
                    while u2 > cap2 and t2:
                        _, m = t2.popitem(last=False)
                        u2 -= m
        return hits / total if total else 0.0
    raise ValueError(policy)


def sweep(trace, gib=(0.25, 0.5, 1, 2, 4)) -> list[dict]:
    rows = []
    for g in gib:
        cap = int(g * 2**30 / KV_BYTES_PER_TOKEN)
        rows.append({"kv_gib": g, "capacity_tokens": cap,
                     **{p: round(simulate(trace, cap, p), 4) for p in ("lru", "lfu", "pinned")}})
    return rows


__all__ = ["collect", "reuse", "simulate", "sweep", "KV_BYTES_PER_TOKEN"]
