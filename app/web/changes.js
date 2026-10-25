/* Kanoon-Bridge: "What changed" page. Reads GET /api/drift/list and GET /api/drift?ref=...
   (drift/explain.py), so it works without the search indexes. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const CLASSES = [
    ["all", "All"], ["substantive", "Provision changed"], ["punishment", "Punishment changed"],
    ["restructured", "Merged / split"], ["cosmetic", "Reworded"], ["identical", "Same text"],
    ["new", "New in BNS"], ["repealed", "Dropped"], ["unverified", "Not verified"],
  ];
  let rows = [];
  let filter = "all";

  try {
    const s = JSON.parse(localStorage.getItem("kb-settings") || "{}");
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = s.theme || (dark ? "dark" : "light");
  } catch { /* private mode */ }

  const label = (r) => r.bns ? `BNS ${r.bns.toUpperCase()}` : `IPC ${r.ipc.toUpperCase()}`;

  function renderFilters() {
    const counts = rows.reduce((m, r) => (m[r.class] = (m[r.class] || 0) + 1, m), {});
    $("#ch-filters").innerHTML = CLASSES.map(([k, t]) =>
      `<button type="button" class="chip" data-k="${k}" aria-pressed="${k === filter}">${esc(t)}` +
      ` <span class="n">${k === "all" ? rows.length : counts[k] || 0}</span></button>`).join("");
  }

  function renderList() {
    const shown = rows.filter((r) => filter === "all" || r.class === filter);
    $("#ch-list").innerHTML = shown.map((r) => `
      <button type="button" class="ch-row" data-ref="${r.bns ? "bns:" + r.bns : "ipc:" + r.ipc}">
        <span class="ch-ref">${esc(label(r))}</span>
        <span class="ch-title">${esc(r.title || "")}<span class="ch-from">${r.bns && r.ipc ? " ← IPC " + esc(r.ipc.toUpperCase().replaceAll(";", ", ")) : ""}</span></span>
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
    const olds = v.ipc_sections.map((s) => `IPC ${s.toUpperCase()}`).join(", ") || "—";
    const pct = Math.round(v.carry_over * 100);
    let html = `
      <h2>${esc(olds)} <span class="arrow">→</span> ${v.bns_section ? "BNS " + esc(v.bns_section.toUpperCase()) : "no BNS successor"}</h2>
      ${v.bns_title ? `<p class="ch-sub">${esc(v.bns_title)}</p>` : ""}
      <div class="ch-verdict t-${v.class}"><strong>${esc(v.class_text)}</strong>
        <div class="co"><span>Carry-over of IPC-era precedent</span>
        <span class="meter big"><span style="width:${pct}%"></span></span><b>${v.carry_over.toFixed(2)}</b></div></div>
      <ul class="ch-reasons">${v.reasons.slice(v.bns_section ? 1 : 0).map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`;
    if (v.punishment_old || v.punishment_new) {
      html += `<table class="ch-pun"><tr><th>IPC punishment</th><td>${esc(v.punishment_old || "—")}</td></tr>
               <tr><th>BNS punishment</th><td>${esc(v.punishment_new || "—")}</td></tr></table>`;
    }
    if ((v.pairs || []).length > 1) {
      html += `<h3>Each IPC predecessor</h3><ul class="ch-reasons">` + v.pairs.map((p) =>
        `<li>IPC ${esc(p.ipc.toUpperCase())}: ${esc(p.class)}, carry-over ${p.carry_over.toFixed(2)}</li>`).join("") + `</ul>`;
    }
    if (v.diff) {
      const [l, r] = diffHtml(v.diff);
      html += `<h3>Text, word by word</h3><p class="ch-legend"><del>removed</del> <ins>added</ins> <span class="d-ren">renumbered reference</span></p>
        <div class="ch-diff"><div><h4>IPC</h4><p>${l}</p></div><div><h4>BNS</h4><p>${r}</p></div></div>`;
    } else if (v.bns_section) {
      html += `<p class="ch-note">The side-by-side text needs the bare acts on disk (<code>make fetch</code>).</p>`;
    }
    box.innerHTML = html;
    if (scroll && window.innerWidth <= 860) box.scrollIntoView({ block: "start", behavior: "smooth" });
  }

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
    renderFilters(); renderList();
    const ref = new URLSearchParams(location.search).get("ref");
    show(ref || "ipc:124a", false);
  }).catch(() => { $("#ch-list").innerHTML = `<p class="error">Could not load the drift table.</p>`; });
})();
