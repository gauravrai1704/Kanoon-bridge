/* Kanoon-Bridge single-page app. Talks to app/web_server.py (POST /api/search, GET /api/similar,
   GET /api/doc, GET /api/health). History and settings live in this browser (localStorage). */

(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const STATES = ["andaman-nicobar", "andhra-pradesh", "arunachal-pradesh", "assam", "bihar", "chandigarh", "chhattisgarh",
    "dadra-nagar-haveli-daman-diu", "delhi", "goa", "gujarat", "haryana", "himachal-pradesh", "jammu-kashmir", "jharkhand",
    "karnataka", "kerala", "ladakh", "lakshadweep", "madhya-pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "puducherry", "punjab", "rajasthan", "sikkim", "tamil-nadu", "telangana", "tripura",
    "uttar-pradesh", "uttarakhand", "west-bengal"];
  const pretty = (slug) => slug.split("-").map((w) => w[0].toUpperCase() + w.slice(1)).join(" ");
  const EXAMPLES = [
    { q: "mere bhai ko chaku maara", state: "delhi", date: "2025-03-01" },
    { q: "dowry death and cruelty by husband", state: "uttar-pradesh", date: "2023-02-10" },
    { q: "BNS 103 knife", state: "maharashtra", date: "2025-01-03" },
    { q: "punishmnt for murdr", state: "", date: "2025-01-03" },
    { q: "\"criminal breach of trust\" AND NOT acquittal", state: "", date: "" },
    { q: "anticipatory bail after:2015", state: "delhi", date: "" },
  ];
  const ADVOCATES = { meera: { name: "Adv. Meera Rao" }, kabir: { name: "Adv. Kabir Sethi" } };

  // ------------------------------------------------------------------ storage (per browser)
  const store = {
    get(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode: keep going */ } },
  };
  const settings = Object.assign({ advocate: "meera", theme: null, state: "", answer: true, agent: false }, store.get("kb-settings", {}));
  let history = store.get("kb-history", []);
  let current = null;           // last response
  let currentReq = null;        // last request body
  let marked = new Set();       // relevance feedback

  // ------------------------------------------------------------------ setup
  function paintIcons(root = document) {
    $$("[data-icon]", root).forEach((el) => { el.innerHTML = ART.icon[el.dataset.icon] || ""; });
  }
  function applyTheme() {
    const sysDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const theme = settings.theme || (sysDark ? "dark" : "light");
    document.documentElement.dataset.theme = theme;
    $("#theme-icon").innerHTML = theme === "dark" ? ART.icon.sun : ART.icon.moon;
    $("#theme-label").textContent = theme === "dark" ? "Light" : "Dark";
  }
  function saveSettings() { store.set("kb-settings", settings); }

  function init() {
    $("#brand-logo").innerHTML = ART.logo(34);
    $("#top-logo").innerHTML = ART.logo(26);
    $("#bridge-art").innerHTML = ART.bridge();
    paintIcons();
    applyTheme();

    $("#state").innerHTML = `<option value="">Not specified</option>` + STATES.map((s) => `<option value="${s}">${pretty(s)}</option>`).join("");
    $("#state").value = settings.state || "";
    $("#opt-answer").checked = !!settings.answer;
    $("#opt-agent").checked = !!settings.agent;
    $("#date").value = new Date().toISOString().slice(0, 10);

    $("#examples").innerHTML = EXAMPLES.map((e, i) => `<button type="button" class="example" data-i="${i}">${esc(e.q)}</button>`).join("");
    $("#examples").addEventListener("click", (ev) => {
      const b = ev.target.closest(".example"); if (!b) return;
      const e = EXAMPLES[+b.dataset.i];
      $("#q").value = e.q; $("#state").value = e.state; if (e.date) $("#date").value = e.date; else $("#date").value = "";
      submit();
    });

    $$("[data-adv-art]").forEach((el) => { el.innerHTML = ART.advocate(el.dataset.advArt); });
    $$(".adv-choice").forEach((b) => b.addEventListener("click", () => { settings.advocate = b.dataset.adv; saveSettings(); markAdvocate(); if (current) renderAdvocate(current); }));
    markAdvocate();

    $("#search-form").addEventListener("submit", (ev) => { ev.preventDefault(); submit(); });
    $("#state").addEventListener("change", () => { settings.state = $("#state").value; saveSettings(); });
    $("#opt-answer").addEventListener("change", () => { settings.answer = $("#opt-answer").checked; saveSettings(); });
    $("#opt-agent").addEventListener("change", () => { settings.agent = $("#opt-agent").checked; saveSettings(); });
    $("#theme-toggle").addEventListener("click", () => {
      settings.theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; saveSettings(); applyTheme();
    });
    $("#new-q").addEventListener("click", newQuestion);
    $("#clear-history").addEventListener("click", () => { history = []; store.set("kb-history", history); renderHistory(); });
    $("#menu-btn").addEventListener("click", () => toggleMenu(true));
    $("#scrim").addEventListener("click", () => toggleMenu(false));
    $("#hood-btn").addEventListener("click", openHood);
    $("#hood-close").addEventListener("click", closeHood);
    $("#hood-replay").addEventListener("click", () => animateRun(true));
    $("#hood-raw").addEventListener("click", toggleRaw);
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") { closeHood(); toggleMenu(false); }
      if (ev.key === "/" && document.activeElement.tagName !== "INPUT") { ev.preventDefault(); $("#q").focus(); }
    });
    window.addEventListener("hashchange", fromHash);

    renderHistory();
    health();
    fromHash();
  }

  function markAdvocate() {
    $$(".adv-choice").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.adv === settings.advocate)));
  }
  function toggleMenu(open) {
    document.body.classList.toggle("menu-open", open);
    $("#scrim").hidden = !open;
  }

  async function health() {
    try {
      const h = await (await fetch("api/health")).json();
      const f = h.features;
      $("#status-line").textContent = `${h.statutes.toLocaleString()} statutes and ${h.precedents.toLocaleString()} judgments indexed. `
        + `${f.claude ? "Answers by Claude." : "Answers quoted from sources (add an API key for Claude)."}`;
    } catch {
      $("#status-line").textContent = "The search server is not running. Start it with: make web";
    }
  }

  // ------------------------------------------------------------------ searching
  function newQuestion() {
    current = null; marked = new Set();
    document.body.classList.remove("has-results");
    $("#results").hidden = true; $("#error").hidden = true; $("#hood-btn").hidden = true;
    $("#q").value = ""; history.forEach((h) => (h.active = false));
    $("#bridge-art").innerHTML = ART.bridge();
    location.hash = "";
    renderHistory(); toggleMenu(false); closeHood();
    $("#q").focus();
  }

  function submit(extra = {}) {
    const q = $("#q").value.trim();
    if (!q) { $("#q").focus(); return; }
    const body = {
      q, state: $("#state").value, date: $("#date").value,
      answer: $("#opt-answer").checked, agent: $("#opt-agent").checked, k: 10, ...extra,
    };
    if (!extra.feedback) marked = new Set();
    run(body);
  }

  async function run(body) {
    currentReq = body;
    $("#error").hidden = true;
    $("#results").hidden = true;
    $("#loading").hidden = false;
    $("#crossing").innerHTML = `<span class="tag l">IPC</span><span class="tag r">BNS</span><span class="lamp"></span>`;
    $("#loading-text").textContent = body.agent ? "The research agent is running several searches…" : "Crossing from IPC to BNS…";
    if (!document.body.classList.contains("has-results")) { document.body.classList.add("has-results"); $("#bridge-art").innerHTML = ART.strip(); }
    toggleMenu(false);
    try {
      const res = await fetch("api/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      current = data;
      render(data);
      remember(body, data);
      const hash = new URLSearchParams({ q: body.q, state: body.state || "", date: body.date || "" }).toString();
      if (location.hash.slice(1) !== hash) history_replace(hash);
    } catch (err) {
      $("#error").textContent = String(err.message || err).includes("Failed to fetch")
        ? "Cannot reach the search server. Start it with: make web" : String(err.message || err);
      $("#error").hidden = false;
    } finally {
      $("#loading").hidden = true;
    }
  }
  function history_replace(hash) { window.removeEventListener("hashchange", fromHash); location.hash = hash; setTimeout(() => window.addEventListener("hashchange", fromHash), 0); }
  function fromHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (!p.get("q")) return;
    $("#q").value = p.get("q"); $("#state").value = p.get("state") || ""; $("#date").value = p.get("date") || "";
    submit();
  }

  // ------------------------------------------------------------------ history
  function remember(body, data) {
    const lead = leadLine(data);
    history = history.filter((h) => !(h.q === body.q && h.state === body.state && h.date === body.date));
    history.unshift({ ...body, feedback: undefined, ts: Date.now(), lead });
    history = history.slice(0, 60);
    history.forEach((h, i) => (h.active = i === 0));
    store.set("kb-history", history);
    renderHistory();
  }
  function renderHistory() {
    const list = $("#history-list");
    $("#history-empty").hidden = history.length > 0;
    const today = new Date().toDateString();
    const groups = { Today: [], Earlier: [] };
    history.forEach((h, i) => (new Date(h.ts).toDateString() === today ? groups.Today : groups.Earlier).push([h, i]));
    list.innerHTML = Object.entries(groups).filter(([, v]) => v.length).map(([name, items]) => `
      <p class="history-group">${name}</p>
      ${items.map(([h, i]) => `<button type="button" class="history-item" data-i="${i}" ${h.active ? 'aria-current="true"' : ""}>
          <span class="h-q">${esc(h.q)}</span>
          <span class="h-meta">${esc([h.state ? pretty(h.state) : null, h.date || null, h.agent ? "agent" : null].filter(Boolean).join(", ") || "no state or date")}</span>
        </button>`).join("")}`).join("");
    list.onclick = (ev) => {
      const b = ev.target.closest(".history-item"); if (!b) return;
      const h = history[+b.dataset.i];
      $("#q").value = h.q; $("#state").value = h.state || ""; $("#date").value = h.date || "";
      $("#opt-answer").checked = !!h.answer; $("#opt-agent").checked = !!h.agent;
      submit();
    };
  }

  // ------------------------------------------------------------------ rendering
  function render(d) {
    $("#bridge-art").innerHTML = ART.strip();
    $("#results").hidden = false;
    $("#hood-btn").hidden = false;
    renderUnderstood(d);
    renderAdvocate(d);
    renderCases(d);
    renderLaw(d);
    renderFeedbackBar();
    prepareHood(d);
    window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
  }

  function renderUnderstood(d) {
    const u = d.understanding;
    const lang = { en: "English", hi: "Hindi", hinglish: "Hinglish" }[u.language] || u.language;
    const parts = [];
    if (u.did_you_mean) {
      parts.push(`<p class="dym">Showing results for corrected spelling. Did you mean <button type="button" id="dym">${esc(u.did_you_mean)}</button>?</p>`);
    }
    parts.push(`<span class="chip">Read as <strong>${esc(lang)}</strong></span>`);
    if (u.normalised) parts.push(`<span class="chip">Understood as <strong>${esc(u.normalised)}</strong></span>`);
    if (u.code_in_force !== "unknown") parts.push(`<span class="chip code-${u.code_in_force}">Code in force <strong>${u.code_in_force.toUpperCase()}</strong></span>`);
    u.crossings.forEach((c) => parts.push(`<span class="chip crossing-chip" title="${esc(c.offence)}"><span>${esc(c.from)}</span>${ART.span}<strong>${esc(c.to)}</strong></span>`));
    if (!u.crossings.length) u.offences.slice(0, 2).forEach((o) => parts.push(`<span class="chip">Offence <strong>${esc(o)}</strong></span>`));
    if (u.boolean) parts.push(`<span class="chip">Boolean <strong>${esc(u.boolean)}</strong></span>`);
    Object.entries(u.wildcards || {}).forEach(([p, t]) => parts.push(`<span class="chip">${esc(p)} matched <strong>${t.length} terms</strong></span>`));
    if (d.state) parts.push(`<span class="chip">Binding courts for <strong>${esc(pretty(d.state))}</strong></span>`);
    if (d.feedback) parts.push(`<span class="chip">Refined with your feedback <strong>+${d.feedback.added.length} terms</strong></span>`);
    $("#understood").innerHTML = parts.join("");
    const dym = $("#dym");
    if (dym) dym.onclick = () => { $("#q").value = u.did_you_mean; submit(); };
  }

  function leadLine(d) {
    const s = d.statutes.find((x) => x.in_force) || d.statutes[0];
    return s ? `${s.ref}: ${s.title}` : (d.precedents[0] ? d.precedents[0].title : "No match");
  }

  function brief(d) {
    const u = d.understanding;
    const lines = [];
    if (d.date && u.code_in_force !== "unknown") {
      lines.push(`For an incident on ${fmtDate(d.date)}, the ${u.code_in_force === "bns" ? "Bharatiya Nyaya Sanhita (BNS)" : "Indian Penal Code (IPC)"} applies.`);
    }
    u.crossings.slice(0, 2).forEach((c) => lines.push(`${c.from} and ${c.to} are the same offence: ${c.offence.toLowerCase()}.`));
    const s = d.statutes.find((x) => x.in_force) || d.statutes[0];
    if (s) lines.push(`The closest provision is ${s.ref}, “${s.title}”.${s.version && /new in BNS/.test(s.version) ? " It is new in the BNS." : ""}`);
    const binding = d.precedents.find((p) => p.status === "binding");
    const top = binding || d.precedents[0];
    if (top) lines.push(`${binding ? `The strongest judgment that binds ${d.state ? pretty(d.state) : "everywhere"}` : "The closest judgment"} is ${top.title} (${[top.court, top.year].filter(Boolean).join(", ")}).`);
    return lines;
  }
  function fmtDate(iso) {
    try { return new Date(iso + "T00:00:00").toLocaleDateString("en-IN", { day: "numeric", month: "long", year: "numeric" }); } catch { return iso; }
  }

  function renderAdvocate(d) {
    const who = settings.advocate;
    const a = d.answer;
    let mood = "idle", lead = leadLine(d), body = "", foot = "";
    if (a && !a.error && !a.abstained && a.sentences.length) {
      mood = "talking";
      body = `<p>${a.sentences.map((s) => `<span class="sent">${esc(s.text)}</span>${s.cite ? `<button type="button" class="cite" data-n="${s.cite}" aria-label="Source ${s.cite}">${s.cite}</button>` : ""}`).join(" ")}</p>`;
      body += `<div class="sources">${a.sources.map((s) => `<span><b>${s.n}</b>${esc(s.title)}</span>`).join("")}</div>`;
      if (a.flags.length) body += `<div class="notes">${a.flags.map((f) => note(f)).join("")}</div>`;
      foot = `<span>Written by ${a.generator === "claude" ? "Claude" : "quoting the sources directly"}; ${Math.round((a.supported || 0) * 100)}% of sentences are backed by the source they cite.</span>`;
    } else {
      const lines = brief(d);
      mood = a && a.abstained ? "unsure" : "happy";
      if (a && a.abstained) body += `<p>I would rather not write an answer for this one: ${esc(a.text.replace(/^Not enough grounding in the indexed law to answer this \(|\)\.?$/g, ""))}. Here is what the search found.</p>`;
      if (a && a.error) body += `<p>${esc(a.error)}</p>`;
      body += lines.length ? `<p>${lines.map(esc).join(" ")}</p>` : `<p>Nothing in the indexed law matched. Try a section number, fewer words, or a different date.</p>`;
      if (!a) foot = `<span>Turn on “Advocate's answer” for a written, cited answer.</span>`;
    }
    foot += `<button type="button" class="ghost" id="adv-hood"><span class="ic" data-icon="gears"></span>How this was found</button>`;
    $("#advocate-panel").innerHTML = `
      <div class="adv-figure">${ART.advocate(who, mood)}
        <div><div class="adv-name">${ADVOCATES[who].name}</div><div class="adv-role">your Kanoon-Bridge advocate</div></div>
      </div>
      <div class="bubble">
        <p class="bubble-lead">${esc(lead)}</p>
        ${body}
        <div class="bubble-foot">${foot}</div>
      </div>`;
    paintIcons($("#advocate-panel"));
    $("#adv-hood").onclick = openHood;
    $$(".cite", $("#advocate-panel")).forEach((b) => (b.onclick = () => {
      const src = a.sources.find((s) => s.n === +b.dataset.n);
      if (!src) return;
      const target = document.querySelector(`[data-doc="${CSS.escape(src.id)}"]`);
      if (target) { target.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "center" }); target.animate?.([{ background: "var(--mark)" }, { background: "transparent" }], 1400); }
      else openDoc(src.id);
    }));
    if (mood === "talking") setTimeout(() => { const svg = $(".advocate", $("#advocate-panel")); if (svg) svg.classList.replace("mood-talking", "mood-happy"); }, 2400);
  }

  function note(flag) {
    const bare = flag.match(/^bare number:\s*(\S+)\s*->\s*(\w+):(\S+)\s+([\d.]+)/);
    if (bare) {
      return `<div class="note"><span class="tape">Ambiguous number</span><span>“${esc(bare[1])}” has no code in the answer; most likely ${esc(bare[2].toUpperCase())} ${esc(bare[3])} (${Math.round(+bare[4] * 100)}%).</span></div>`;
    }
    const m = flag.match(/^(unsupported|wrong code|bare number)[^:]*:\s*(.*)$/s);
    const label = { unsupported: "Weak support", "wrong code": "Check the code", "bare number": "Ambiguous number" }[m ? m[1] : ""] || "Check";
    return `<div class="note"><span class="tape">${label}</span><span>${esc(m ? m[2] : flag)}</span></div>`;
  }

  function renderCases(d) {
    if (!d.precedents.length) { $("#cases").innerHTML = `<p class="empty">No judgments matched. Try fewer words, a section number, or remove filters.</p>`; return; }
    $("#cases").innerHTML = d.precedents.map((p) => `
      <article class="case" data-doc="${esc(p.id)}">
        <button type="button" class="case-title" data-open="${esc(p.id)}">${esc(p.title)}</button>
        <div class="case-meta">
          <span>${esc(p.court || "Court not recorded")}${p.year ? `, ${p.year}` : ""}</span>
          ${p.status === "binding" ? `<span class="badge binding"><span class="ic" data-icon="scales"></span>${d.state ? "Binds " + esc(pretty(d.state)) : "Binding"}</span>` : ""}
          ${p.status === "persuasive" ? `<span class="badge persuasive">Persuasive only</span>` : ""}
          ${p.code && p.code !== "?" ? `<span>decided under ${esc(p.code.toUpperCase())}</span>` : ""}
        </div>
        ${p.snippet ? `<p class="case-snippet">${p.snippet}</p>` : ""}
        ${p.why.length ? `<ul class="why">${p.why.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}
        ${p.found_by.length ? `<p class="found-by">Found by the agent's ${p.found_by.map((f) => f.split(":")[1].replace("_", " ")).join(", ")} searches</p>` : ""}
        <div class="case-actions">
          <button type="button" class="ghost" data-mark="${esc(p.id)}" aria-pressed="${marked.has(p.id)}">${marked.has(p.id) ? "Marked relevant" : "Mark relevant"}</button>
          <button type="button" class="ghost" data-similar="${esc(p.id)}">Similar cases</button>
          <button type="button" class="ghost" data-score="${esc(p.id)}">Score details</button>
        </div>
        <div class="case-extra"></div>
      </article>`).join("");
    paintIcons($("#cases"));
    $("#cases").onclick = async (ev) => {
      const t = ev.target.closest("button"); if (!t) return;
      const art = t.closest(".case"); const extra = $(".case-extra", art);
      if (t.dataset.open) openDoc(t.dataset.open);
      if (t.dataset.mark) {
        const id = t.dataset.mark;
        marked.has(id) ? marked.delete(id) : marked.add(id);
        t.setAttribute("aria-pressed", String(marked.has(id))); t.textContent = marked.has(id) ? "Marked relevant" : "Mark relevant";
        renderFeedbackBar();
      }
      if (t.dataset.score) {
        const p = d.precedents.find((x) => x.id === t.dataset.score);
        extra.innerHTML = extra.innerHTML ? "" : `<div class="breakdown">${Object.entries(p.components).map(([k, v]) => `${esc(k)} ${v}`).join("<br>")}</div>`;
      }
      if (t.dataset.similar) {
        if (extra.innerHTML) { extra.innerHTML = ""; return; }
        extra.innerHTML = `<div class="similar">Finding similar cases…</div>`;
        try {
          const s = await (await fetch(`api/similar?id=${encodeURIComponent(t.dataset.similar)}&k=5`)).json();
          extra.innerHTML = `<div class="similar">Cases like this one, by text, shared offences (in either code) and co-citation:
            <ol>${s.similar.map((x) => `<li><button type="button" class="case-title" style="font-size:1rem" data-open="${esc(x.id)}">${esc(x.title)}</button>
              <span class="parts">text ${x.parts.text.toFixed(2)}, shared offences ${x.parts.coupling.toFixed(2)}, cited together ${x.parts.cocitation.toFixed(2)}</span></li>`).join("") || "<li>None found.</li>"}</ol></div>`;
        } catch { extra.innerHTML = `<div class="similar">Could not load similar cases.</div>`; }
      }
    };
  }

  function renderFeedbackBar() {
    const bar = $("#feedback-bar");
    bar.hidden = marked.size === 0;
    bar.innerHTML = `<span>${marked.size} judgment${marked.size === 1 ? "" : "s"} marked relevant. Refining moves the search toward them.</span>
      <button type="button" class="go" id="refine">Refine with feedback</button>`;
    const r = $("#refine"); if (r) r.onclick = () => submit({ feedback: [...marked] });
  }

  function renderLaw(d) {
    if (!d.statutes.length) { $("#law").innerHTML = `<p class="empty">No statute matched.</p>`; return; }
    $("#law").innerHTML = d.statutes.slice(0, 6).map((s) => `
      <article class="statute${s.in_force ? " in-force" : ""}">
        <div class="statute-ref"><span>${esc(s.ref)}</span>${s.in_force ? `<span class="force">in force on your date</span>` : ""}</div>
        <div class="statute-title">${esc(s.title)}</div>
        ${s.text ? `<p class="statute-text">${esc(s.text)}</p>` : ""}
        ${s.version ? `<div class="statute-version">${esc(s.version.replace("<-", "replaces").replace("->", "became"))}</div>` : ""}
      </article>`).join("");
  }

  // ------------------------------------------------------------------ reader
  async function openDoc(id) {
    const dlg = $("#reader");
    $("#reader-body").innerHTML = `<p>Opening…</p>`;
    dlg.showModal();
    try {
      const doc = await (await fetch(`api/doc?id=${encodeURIComponent(id)}`)).json();
      if (doc.error) throw new Error(doc.error);
      const zoneName = { facts: "Facts", arguments: "Arguments", ratio: "Reasoning", decision: "Decision", statute: "Text", other: "" };
      let last = null;
      $("#reader-body").innerHTML = `<h3>${esc(doc.title || doc.id)}</h3>
        <p class="r-meta">${esc([doc.court, doc.date].filter(Boolean).join(", "))}</p>
        ${doc.paragraphs.map((p) => { const head = p.zone !== last && zoneName[p.zone] ? `<p class="zone">${zoneName[p.zone]}</p>` : ""; last = p.zone; return head + `<p>${esc(p.text)}</p>`; }).join("")}`;
    } catch (err) { $("#reader-body").innerHTML = `<p class="error">${esc(err.message)}</p>`; }
  }

  // ------------------------------------------------------------------ under the hood
  const TIME_COLOURS = ["var(--brass)", "var(--bind)", "var(--tape)", "var(--persuade)", "var(--ink-3)"];
  const GROUP_COLOURS = { "Understand": "var(--ink-3)", "Find the law": "var(--brass)", "Find cases": "var(--bind)", "Research agent": "var(--persuade)", "Answer": "var(--tape)" };
  function prepareHood(d) {
    const t = d.timings_ms || {};
    const parts = Object.entries(t).filter(([k]) => k !== "search" || Object.keys(t).length === 1);
    const total = parts.reduce((a, [, v]) => a + v, 0) || 1;
    $("#hood-sub").textContent = `“${d.query}” took ${Math.round((t.search || 0) + (t.answer || 0))} ms across ${d.pipeline.length} steps.`;
    $("#timebar").innerHTML = parts
      .map(([k, v], i) => `<span title="${esc(k)} ${v} ms" style="width:${(v / total) * 100}%;background:${TIME_COLOURS[i % 5]}"></span>`).join("");
    const TIME_NAMES = { analyze: "understanding", statutes: "statutes", bridge: "bridge", precedents: "judgments", answer: "answer", agent: "agent searches", search: "search" };
    $("#time-legend").innerHTML = parts.map(([k, v], i) => `<span><i style="background:${TIME_COLOURS[i % 5]}"></i>${TIME_NAMES[k] || esc(k)} ${Math.round(v)} ms</span>`).join("");
    let lastGroup = null;
    $("#run").innerHTML = d.pipeline.map((s) => {
      const head = s.group !== lastGroup ? `<li class="grp" style="color:${GROUP_COLOURS[s.group] || "var(--brass)"}">${esc(s.group)}</li>` : "";
      lastGroup = s.group;
      return head + `<li class="stage${s.on ? "" : " off"}">
        <div class="stage-name"><span>${esc(s.name)}</span>${s.ms != null ? `<span class="stage-ms">${s.ms} ms</span>` : ""}</div>
        <div class="stage-detail">${esc(s.detail)}</div></li>`;
    }).join("");
    $("#raw").textContent = JSON.stringify(d, null, 2);
  }
  function openHood() {
    if (!current) return;
    $("#hood").hidden = false;
    animateRun(false);
    $("#hood-close").focus();
  }
  function closeHood() { $("#hood").hidden = true; }
  function animateRun() {
    const run = $("#run");
    const stages = $$(".stage", run);
    if (reduced) { stages.forEach((s) => s.classList.add("lit")); return; }
    run.classList.add("animate");
    stages.forEach((s) => s.classList.remove("lit"));
    stages.forEach((s, i) => setTimeout(() => { s.classList.add("lit"); if (i === stages.length - 1) setTimeout(() => run.classList.remove("animate"), 600); }, 120 + i * 110));
  }
  function toggleRaw() {
    const btn = $("#hood-raw"); const on = btn.getAttribute("aria-pressed") !== "true";
    btn.setAttribute("aria-pressed", String(on)); btn.textContent = on ? "Show the steps" : "Show raw response";
    $("#raw").hidden = !on; $("#run").hidden = on; $("#timebar").hidden = on; $("#time-legend").hidden = on;
  }

  init();
})();
