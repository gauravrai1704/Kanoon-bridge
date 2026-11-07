/* Kanoon-Bridge: "What changed" page. Reads GET /api/drift/list and GET /api/drift?ref=...
   (drift/explain.py), so it works without the search indexes. Three code families:
   IPC -> BNS, CrPC -> BNSS, Evidence Act -> BSA. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const CLASSES = [
    ["all", "All"], ["substantive", "Provision changed"], ["punishment", "Punishment changed"],
    ["restructured", "Merged / split"], ["cosmetic", "Reworded"], ["identical", "Same text"],
    ["new", "New section"], ["repealed", "Dropped"], ["unverified", "Not verified"],
  ];
  const CODE_OF = { ipc: ["ipc", "bns"], crpc: ["crpc", "bnss"], iea: ["iea", "bsa"] };
  let rows = [];
  let families = [];
  let family = "ipc";
  let filter = "all";

  try {
    const s = JSON.parse(localStorage.getItem("kb-settings") || "{}");
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = s.theme || (dark ? "dark" : "light");
  } catch { /* private mode */ }

  const label = (r) => r.new ? `${r.new_label} ${r.new.toUpperCase()}` : `${r.old_label} ${r.old.toUpperCase()}`;
  const inFamily = () => rows.filter((r) => r.family === family);

  function renderFamilies() {
    $("#ch-families").innerHTML = families.map((f) =>
      `<button type="button" class="chip fam" data-f="${f.family}" aria-pressed="${f.family === family}">` +
      `${esc(f.old_label)} → ${esc(f.new_label)}</button>`).join("");
  }

  function renderFilters() {
    const mine = inFamily();
    const counts = mine.reduce((m, r) => (m[r.class] = (m[r.class] || 0) + 1, m), {});
    $("#ch-filters").innerHTML = CLASSES.filter(([k]) => k === "all" || counts[k]).map(([k, t]) =>
      `<button type="button" class="chip" data-k="${k}" aria-pressed="${k === filter}">${esc(t)}` +
      ` <span class="n">${k === "all" ? mine.length : counts[k] || 0}</span></button>`).join("");
  }

  function renderList() {
    const shown = inFamily().filter((r) => filter === "all" || r.class === filter);
    $("#ch-list").innerHTML = shown.map((r) => `
      <button type="button" class="ch-row" data-ref="${esc(r.ref)}">
        <span class="ch-ref">${esc(label(r))}</span>
        <span class="ch-title">${esc(r.title || "")}<span class="ch-from">${r.new && r.old ? " ← " + esc(r.old_label) + " " + esc(r.old.toUpperCase().replaceAll(";", ", ")) : ""}</span></span>
        <span class="ch-tag t-${r.class}">${esc(r.class)}</span>
        <span class="meter" title="carry-over ${r.carry_over.toFixed(2)}"><span style="width:${Math.round(r.carry_over * 100)}%"></span></span>
      </button>`).join("") || `<p class="ch-empty">No sections in this group.</p>`;
  }

  function diffHtml(diff) {
    const left = [], right = [];
    for (const d of diff) {
      if (d.op === "equal") { left.push(esc(d.old)); right.push(esc(d.new)); }
      else if (d.op === "renumbered") { left.push(`<span class="d-ren">${esc(d.old)}</span>`); right.push(`<span class="d-ren">${esc(d.new)}</span>`); }
      else {
        if (d.old) left.push(`<del>${esc(d.old)}</del>`);
        if (d.new) right.push(`<ins>${esc(d.new)}</ins>`);
      }
    }
    return [left.join(" "), right.join(" ")];
  }

  async function show(ref, scroll = true) {
    const box = $("#ch-detail");
    box.innerHTML = `<p class="ch-empty">Loading…</p>`;
    let v;
    try {
      const r = await fetch(`/api/drift?ref=${encodeURIComponent(ref)}`);
      v = await r.json();
    } catch (e) { box.innerHTML = `<p class="error">Could not reach the server.</p>`; return; }
    if (v.error || !v.found) { box.innerHTML = `<p class="ch-empty">${esc(v.error || v.message)}</p>`; return; }
    history.replaceState(null, "", `?ref=${encodeURIComponent(v.ref)}`);
    if (v.family && v.family !== family) { family = v.family; filter = "all"; renderFamilies(); renderFilters(); renderList(); }
    const O = v.old_label, N = v.new_label;
    const olds = v.old_sections.map((s) => `${O} ${s.toUpperCase()}`).join(", ") || "—";
    const pct = Math.round(v.carry_over * 100);
    let html = `
      <h2>${esc(olds)} <span class="arrow">→</span> ${v.new_section ? esc(N) + " " + esc(v.new_section.toUpperCase()) : "no " + esc(N) + " successor"}</h2>
      ${v.new_title || v.title ? `<p class="ch-sub">${esc(v.new_title || v.title)}</p>` : ""}
      <div class="ch-verdict t-${v.class}"><strong>${esc(v.class_text)}</strong>
        <div class="co"><span>Carry-over of ${esc(O)}-era precedent</span>
        <span class="meter big"><span style="width:${pct}%"></span></span><b>${v.carry_over.toFixed(2)}</b></div></div>
      <ul class="ch-reasons">${v.reasons.slice(v.new_section ? 1 : 0).map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`;
    if (v.punishment_old || v.punishment_new) {
      html += `<table class="ch-pun"><tr><th>${esc(O)} punishment</th><td>${esc(v.punishment_old || "—")}</td></tr>
               <tr><th>${esc(N)} punishment</th><td>${esc(v.punishment_new || "—")}</td></tr></table>`;
    }
    if ((v.pairs || []).length > 1) {
      html += `<h3>Each ${esc(O)} predecessor</h3><ul class="ch-reasons">` + v.pairs.map((p) =>
        `<li>${esc(O)} ${esc(p.old.toUpperCase())}: ${esc(p.class)}, carry-over ${p.carry_over.toFixed(2)}</li>`).join("") + `</ul>`;
    }
    if (v.diff) {
      const [l, r] = diffHtml(v.diff);
      html += `<h3>Text, word by word</h3><p class="ch-legend"><del>removed</del> <ins>added</ins> <span class="d-ren">renumbered reference</span></p>
        <div class="ch-diff"><div><h4>${esc(O)}</h4><p>${l}</p></div><div><h4>${esc(N)}</h4><p>${r}</p></div></div>`;
    } else if (v.new_section) {
      html += `<p class="ch-note">The side-by-side text needs the bare acts on disk (<code>make fetch</code>).</p>`;
    }
    box.innerHTML = html;
    if (scroll && window.innerWidth <= 860) box.scrollIntoView({ block: "start", behavior: "smooth" });
  }

  $("#ch-families").addEventListener("click", (e) => {
    const b = e.target.closest(".chip");
    if (!b) return;
    family = b.dataset.f; filter = "all";
    renderFamilies(); renderFilters(); renderList();
    const first = inFamily().find((r) => r.class === "substantive") || inFamily()[0];
    if (first) show(first.ref, false);
  });
  $("#ch-filters").addEventListener("click", (e) => {
    const b = e.target.closest(".chip");
    if (!b) return;
    filter = b.dataset.k; renderFilters(); renderList();
  });
  $("#ch-list").addEventListener("click", (e) => {
    const b = e.target.closest(".ch-row");
    if (b) show(b.dataset.ref);
  });
  $("#ch-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("#ch-q").value.trim();
    if (q) show(q);
  });

  fetch("/api/drift/list").then((r) => r.json()).then((d) => {
    rows = d.rows || [];
    families = d.families || [];
    const ref = new URLSearchParams(location.search).get("ref") || "ipc:124a";
    const code = ref.split(":")[0].toLowerCase();
    family = Object.keys(CODE_OF).find((f) => CODE_OF[f].includes(code)) || "ipc";
    renderFamilies(); renderFilters(); renderList();
    show(ref, false);
  }).catch(() => { $("#ch-list").innerHTML = `<p class="error">Could not load the drift tables.</p>`; });
})();
