/* Kanoon-Bridge artwork: logo, the bridge hero, the two advocates, small icons.
   All original drawings, inline SVG so they follow the theme colours (currentColor / CSS vars). */

const ART = (() => {
  const logo = (size = 36) => `
  <svg class="logo-mark" width="${size}" height="${size}" viewBox="0 0 48 48" aria-hidden="true">
    <path d="M7 22 Q24 6 41 22" fill="none" stroke="var(--brass)" stroke-width="2.6" stroke-linecap="round"/>
    <line x1="24" y1="11" x2="24" y2="40" stroke="var(--brass)" stroke-width="2.6" stroke-linecap="round"/>
    <line x1="17" y1="40.5" x2="31" y2="40.5" stroke="var(--brass)" stroke-width="2.6" stroke-linecap="round"/>
    <path d="M7 22 L3.5 31 M7 22 L10.5 31 M41 22 L37.5 31 M41 22 L44.5 31" stroke="var(--ink-2)" stroke-width="1.4"/>
    <path d="M2.5 31 Q7 37 11.5 31 Z M36.5 31 Q41 37 45.5 31 Z" fill="var(--brass)"/>
  </svg>`;

  /* The hero bridge: two court pillars (IPC 1860, BNS 2023), a suspension cable, hangers,
     and a deck the search box sits on. Drawn with a 1200 x 380 viewBox. */
  function bridge({ compact = false } = {}) {
    const deckY = compact ? 250 : 262;
    const L = 210, R = 990, top = 64;
    // quadratic cable from (L, top) through control (600, 2*deckY - top - 40) to (R, top)
    const cy = deckY + 150;
    const q = (t) => ({ x: (1 - t) ** 2 * L + 2 * (1 - t) * t * 600 + t * t * R, y: (1 - t) ** 2 * top + 2 * (1 - t) * t * cy + t * t * top });
    let hangers = "";
    for (let i = 1; i < 26; i++) {
      const p = q(i / 26);
      if (p.y < deckY - 8) hangers += `<line x1="${p.x.toFixed(1)}" y1="${p.y.toFixed(1)}" x2="${p.x.toFixed(1)}" y2="${deckY}" class="hanger" style="--i:${i}"/>`;
    }
    const pillar = (x, label, year, side) => `
      <g class="pillar pillar-${side}">
        <rect x="${x - 46}" y="${top - 22}" width="92" height="14" rx="2" class="pillar-cap"/>
        <rect x="${x - 38}" y="${top - 8}" width="76" height="10" class="pillar-cap2"/>
        <rect x="${x - 32}" y="${top + 2}" width="64" height="${360 - top}" class="pillar-shaft"/>
        ${[-20, -7, 7, 20].map((d) => `<line x1="${x + d}" y1="${top + 8}" x2="${x + d}" y2="352" class="flute"/>`).join("")}
        <rect x="${x - 50}" y="${deckY + 44}" width="100" height="58" rx="3" class="plaque"/>
        <text x="${x}" y="${deckY + 74}" class="plaque-code">${label}</text>
        <text x="${x}" y="${deckY + 93}" class="plaque-year">${year}</text>
      </g>`;
    return `
    <svg class="bridge-art${compact ? " compact" : ""}" viewBox="0 0 1200 380" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      <defs>
        <linearGradient id="kb-water" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="var(--water-1)"/><stop offset="1" stop-color="var(--water-2)"/>
        </linearGradient>
        <linearGradient id="kb-fade" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".12" stop-color="#fff"/>
          <stop offset=".88" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity="0"/>
        </linearGradient>
        <mask id="kb-water-mask"><rect x="0" y="0" width="1200" height="380" fill="url(#kb-fade)"/></mask>
      </defs>
      <rect x="0" y="${deckY + 20}" width="1200" height="${380 - deckY}" fill="url(#kb-water)" class="water" mask="url(#kb-water-mask)"/>
      ${[0, 1, 2].map((k) => `<path class="ripple" d="M${60 + k * 30} ${deckY + 60 + k * 22} q 40 -8 80 0 t 80 0 t 80 0 M${760 - k * 40} ${deckY + 70 + k * 18} q 40 -8 80 0 t 80 0 t 80 0"/>`).join("")}
      <path class="cable side" d="M40 ${deckY} Q ${L - 60} ${top + 60} ${L} ${top}"/>
      <path class="cable side" d="M1160 ${deckY} Q ${R + 60} ${top + 60} ${R} ${top}"/>
      <path class="cable main" d="M${L} ${top} Q 600 ${cy} ${R} ${top}"/>
      <g class="hangers">${hangers}</g>
      ${pillar(L, "IPC", "1860", "l")}
      ${pillar(R, "BNS", "2023", "r")}
      <rect x="30" y="${deckY}" width="1140" height="12" rx="3" class="deck"/>
      <circle class="lamp" cx="${L + 50}" cy="${deckY - 7}" r="6"/>
    </svg>`;
  }

  /* The slim version shown above results: one cable, hangers, a deck, the two codes at its ends. */
  function strip() {
    const W = 1000, deck = 70, top = 14;
    let hangers = "";
    for (let x = 140; x <= 860; x += 36) {
      const t = (x - 90) / 820;
      const y = (1 - t) ** 2 * top + 2 * (1 - t) * t * 112 + t * t * top;
      if (y < deck - 6) hangers += `<line class="hanger" x1="${x}" y1="${y.toFixed(1)}" x2="${x}" y2="${deck}"/>`;
    }
    return `<svg class="strip-art" viewBox="0 0 ${W} 92" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      <path class="post" d="M90 4 V${deck} M910 4 V${deck}"/>
      <path class="cable" d="M90 ${top} Q 500 112 910 ${top}"/>
      ${hangers}
      <rect class="deck" x="30" y="${deck}" width="940" height="5" rx="2"/>
      <text class="tag" x="100" y="${deck - 8}">IPC</text><text class="tag r" x="900" y="${deck - 8}">BNS</text>
    </svg>`;
  }

  /* Two original advocate caricatures in Indian court dress: black coat, white bands. */
  function advocate(who = "meera", mood = "idle") {
    const skin = who === "meera" ? "#C68B62" : "#B07850";
    const hair = "#1E1B22";
    const back = who === "meera"
      ? `<circle cx="86" cy="34" r="12" fill="${hair}"/>`
      : "";
    const hairTop = who === "meera"
      ? `<path d="M33 58 C29 30 47 21 62 23 C80 25 91 38 87 60 C83 43 73 35 59 35 C46 35 38 44 33 58 Z" fill="${hair}"/>
         <path d="M40 40 C50 33 62 32 74 37" stroke="#3a3540" stroke-width="1.4" fill="none"/>`
      : `<path d="M34 50 C33 27 50 21 63 23 C79 24 89 35 86 52 C81 41 71 36 58 38 C48 38 40 43 34 50 Z" fill="${hair}"/>
         <path d="M58 38 C63 31 72 29 80 33" stroke="#3a3540" stroke-width="1.4" fill="none"/>`;
    const face = who === "meera"
      ? `<circle cx="34" cy="67" r="2.6" class="adv-earring"/><circle cx="86" cy="67" r="2.6" class="adv-earring"/>`
      : `<path d="M47 69 Q53 64.5 60 67.5 Q67 64.5 73 69 Q66 72.5 60 70.5 Q54 72.5 47 69 Z" fill="${hair}"/>`;
    const specs = who === "kabir"
      ? `<g class="adv-specs" fill="none" stroke="#2a2a33" stroke-width="1.8">
           <circle cx="50" cy="56" r="7.5"/><circle cx="70" cy="56" r="7.5"/><path d="M57.5 55 Q60 53 62.5 55"/></g>`
      : "";
    return `
    <svg class="advocate adv-${who} mood-${mood}" viewBox="0 0 120 140" role="img" aria-label="${who === "meera" ? "Advocate Meera" : "Advocate Kabir"}">
      ${back}
      <path d="M8 140 C12 105 33 92 60 92 C87 92 108 105 112 140 Z" class="adv-coat"/>
      <path d="M47 92 L60 121 L73 92 Z" fill="#F4F4EF"/>
      <path d="M47 92 L56 128 M73 92 L64 128" class="adv-lapel"/>
      <path d="M54.6 101 h4.6 v17 h-4.6 z M60.8 101 h4.6 v17 h-4.6 z" fill="#FFFFFF" stroke="#C9CBD6" stroke-width=".6"/>
      <rect x="52" y="78" width="16" height="16" rx="5" fill="${skin}"/>
      <circle cx="34" cy="58" r="5.5" fill="${skin}"/><circle cx="86" cy="58" r="5.5" fill="${skin}"/>
      <ellipse cx="60" cy="56" rx="26" ry="29" fill="${skin}"/>
      ${hairTop}
      <ellipse cx="45" cy="66" rx="4.5" ry="2.6" fill="#E07A6A" opacity=".28"/>
      <ellipse cx="75" cy="66" rx="4.5" ry="2.6" fill="#E07A6A" opacity=".28"/>
      <g class="adv-brows" stroke="${hair}" stroke-width="2.2" stroke-linecap="round">
        <path d="M44 46.5 Q50 43.5 55 46"/><path d="M65 46 Q70 43.5 76 46.5"/></g>
      <g class="adv-eyes"><circle cx="50" cy="56" r="2.9" fill="#1d1b20"/><circle cx="70" cy="56" r="2.9" fill="#1d1b20"/></g>
      ${specs}
      <path d="M59 58 Q57.5 64 60.5 64.5" stroke="#7a4a33" stroke-width="1.3" fill="none" stroke-linecap="round"/>
      ${face}
      <path class="adv-mouth smile" d="M52 74 Q60 80.5 68 74" stroke="#5a2a22" stroke-width="2" fill="none" stroke-linecap="round"/>
      <path class="adv-mouth flat" d="M53.5 76 Q60 77 66.5 76" stroke="#5a2a22" stroke-width="2" fill="none" stroke-linecap="round"/>
      <ellipse class="adv-mouth talk" cx="60" cy="76" rx="5" ry="3.4" fill="#5a2a22"/>
    </svg>`;
  }

  const icon = {
    scales: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v17M7 20h10M4 8h16M4 8l-2.5 6h5zM20 8l-2.5 6h5z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>`,
    gavel: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13.5 4.5l6 6M11 7l6 6M12.2 5.8l5 5-2.4 2.4-5-5zM10.5 11.5L3.5 18.5M3 21h9" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`,
    pillar: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5h14M6 7h12M8 7v12M12 7v12M16 7v12M5 20h14" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`,
    gears: `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="9" cy="10" r="3" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M9 4v2M9 14v2M3 10h2M13 10h2M4.8 5.8l1.4 1.4M11.8 12.8l1.4 1.4M4.8 14.2l1.4-1.4M11.8 7.2l1.4-1.4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><circle cx="17" cy="17" r="2.2" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M17 13.3v1.2M17 19.5v1.2M13.3 17h1.2M19.5 17h1.2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`,
    plus: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`,
    search: `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="M15.5 15.5L21 21" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
    menu: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`,
    close: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`,
    sun: `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>`,
    moon: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>`,
    link: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12c3-5 11-5 14 0M3 12h3M18 12h3M9 9v6M15 9v6" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`,
    trash: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 7h14M10 7V5h4v2M7 7l1 13h8l1-13" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>`,
  };

  /* A tiny bridge glyph used inside "IPC 302 [bridge] BNS 103" crossing chips. */
  const span = `<svg class="span-glyph" viewBox="0 0 40 16" aria-hidden="true"><path d="M2 13 Q20 -3 38 13" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M1 13h38" stroke="currentColor" stroke-width="1.8"/><path d="M11 6.5v6.5M20 4.5v8.5M29 6.5v6.5" stroke="currentColor" stroke-width="1"/></svg>`;

  return { logo, bridge, strip, advocate, icon, span };
})();
