(function (G) {
  "use strict";

  const API = "/api/detect_ambiguity";

  // ── Colour schemes ─────────────────────────────────────────
  const SC = {
    Good:     { border:"border-green-400 dark:border-green-600",   hbg:"bg-green-50 dark:bg-green-900/20",   bar:"bg-green-500",  badge:"bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300",   icon:"✅" },
    Fair:     { border:"border-amber-400 dark:border-amber-600",   hbg:"bg-amber-50 dark:bg-amber-900/20",   bar:"bg-amber-400",  badge:"bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300",   icon:"⚠️" },
    Poor:     { border:"border-orange-400 dark:border-orange-600", hbg:"bg-orange-50 dark:bg-orange-900/20", bar:"bg-orange-500", badge:"bg-orange-100 dark:bg-orange-900/40 text-orange-700 dark:text-orange-300", icon:"🔶" },
    Critical: { border:"border-red-500 dark:border-red-600",       hbg:"bg-red-50 dark:bg-red-900/20",       bar:"bg-red-500",    badge:"bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300",           icon:"🚨" },
  };
  const SEVC = {
    error:   { ring:"bg-red-100 dark:bg-red-900/30",    txt:"text-red-500",   pill:"bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"    },
    warning: { ring:"bg-amber-100 dark:bg-amber-900/30",txt:"text-amber-500", pill:"bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400" },
    info:    { ring:"bg-blue-100 dark:bg-blue-900/30",  txt:"text-blue-400",  pill:"bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400"   },
  };

  function esc(s){ return String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }
  function deb(fn,ms){ let t; return function(...a){ clearTimeout(t); t=setTimeout(()=>fn(...a),ms); }; }
  function sc(label){ return SC[label]||SC.Fair; }

  // ── API ─────────────────────────────────────────────────────
  async function scanSingle(text) {
    const r = await fetch(API, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({requirement:text}) });
    if (!r.ok) throw new Error("API "+r.status);
    return r.json();
  }
  async function scanBatch(reqs) {
    const r = await fetch(API, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({requirements:reqs}) });
    if (!r.ok) throw new Error("API "+r.status);
    return r.json();
  }

  // ── Inline panel HTML ────────────────────────────────────────
  let _pid = 0;
  function buildInlinePanel(id) {
    return `<div id="${id}" class="ambi-panel hidden rounded-xl border overflow-hidden mt-2 transition-all duration-300">
  <div class="ambi-hdr flex items-center justify-between px-4 py-2.5 cursor-pointer select-none" onclick="Ambiguity._toggle('${id}')">
    <div class="flex items-center gap-2 min-w-0 flex-1">
      <span class="ambi-icon text-sm leading-none"></span>
      <span class="ambi-title text-xs font-bold text-gray-800 dark:text-gray-100"></span>
      <span class="ambi-sbadge text-xs px-1.5 py-0.5 rounded-full font-bold shrink-0"></span>
    </div>
    <div class="flex items-center gap-2 shrink-0 ml-2">
      <span class="ambi-ec hidden text-xs font-bold text-red-500"></span>
      <span class="ambi-wc hidden text-xs font-bold text-amber-500"></span>
      <span class="ambi-ic hidden text-xs font-bold text-blue-400"></span>
      <svg class="ambi-chev w-3.5 h-3.5 text-gray-400 transition-transform duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
    </div>
  </div>
  <div class="ambi-body border-t border-gray-200 dark:border-slate-700">
    <div class="px-4 py-2.5 bg-gray-50 dark:bg-slate-900/40 border-b border-gray-100 dark:border-slate-700">
      <div class="flex items-center justify-between text-xs mb-1">
        <span class="text-gray-400 font-medium">Ambiguity Score</span>
        <span class="ambi-stxt font-bold text-gray-700 dark:text-gray-200"></span>
      </div>
      <div class="w-full bg-gray-200 dark:bg-slate-700 rounded-full h-1.5">
        <div class="ambi-sbar h-1.5 rounded-full transition-all duration-500" style="width:0%"></div>
      </div>
      <div class="flex justify-between text-xs mt-0.5 text-gray-400"><span>Clean</span><span>Critical</span></div>
    </div>
    <div class="ambi-list divide-y divide-gray-100 dark:divide-slate-700/50 max-h-56 overflow-y-auto"></div>
    <div class="ambi-clean hidden py-4 flex flex-col items-center text-center">
      <div class="w-7 h-7 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center mb-1.5">
        <svg class="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7"/></svg>
      </div>
      <p class="text-xs font-bold text-green-600 dark:text-green-400">No issues detected</p>
      <p class="text-xs text-gray-400 mt-0.5">Requirement appears clear and well-formed.</p>
    </div>
  </div>
</div>`;
  }

  function renderInline(report, panelEl) {
    if (!panelEl) return;
    const s = sc(report.quality_label);
    panelEl.className = `ambi-panel rounded-xl border overflow-hidden mt-2 transition-all duration-300 ${s.border}`;
    panelEl.querySelector(".ambi-hdr").className = `ambi-hdr flex items-center justify-between px-4 py-2.5 cursor-pointer select-none ${s.hbg}`;
    panelEl.querySelector(".ambi-icon").textContent = s.icon;
    panelEl.querySelector(".ambi-title").textContent = `Quality: ${report.quality_label}`;
    const sb = panelEl.querySelector(".ambi-sbadge");
    sb.textContent = `${report.ambiguity_score}/100`;
    sb.className = `ambi-sbadge text-xs px-1.5 py-0.5 rounded-full font-bold ${s.badge}`;
    panelEl.querySelector(".ambi-sbar").style.width = report.ambiguity_score + "%";
    panelEl.querySelector(".ambi-sbar").className = `ambi-sbar h-1.5 rounded-full transition-all duration-500 ${s.bar}`;
    panelEl.querySelector(".ambi-stxt").textContent = `${report.ambiguity_score} / 100`;

    const ec = panelEl.querySelector(".ambi-ec");
    const wc = panelEl.querySelector(".ambi-wc");
    const ic = panelEl.querySelector(".ambi-ic");
    report.error_count   > 0 ? (ec.classList.remove("hidden"), ec.textContent = `🔴 ${report.error_count} error${report.error_count>1?"s":""}`)   : ec.classList.add("hidden");
    report.warning_count > 0 ? (wc.classList.remove("hidden"), wc.textContent = `⚠️ ${report.warning_count} warn`)  : wc.classList.add("hidden");
    report.info_count    > 0 ? (ic.classList.remove("hidden"), ic.textContent = `ℹ️ ${report.info_count} note`)     : ic.classList.add("hidden");

    const listEl  = panelEl.querySelector(".ambi-list");
    const cleanEl = panelEl.querySelector(".ambi-clean");
    if (report.warnings && report.warnings.length > 0) {
      cleanEl.classList.add("hidden");
      listEl.innerHTML = report.warnings.map(buildRow).join("");
    } else {
      listEl.innerHTML = "";
      cleanEl.classList.remove("hidden");
    }
  }

  function buildRow(w) {
    const c = SEVC[w.severity] || SEVC.info;
    const ICONS = {
      error:   `<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>`,
      warning: `<path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>`,
      info:    `<path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>`,
    };
    const mb = w.matched_text ? `<code class="mt-0.5 inline-block text-xs font-mono bg-gray-200 dark:bg-slate-700 text-gray-700 dark:text-gray-300 px-1 py-0.5 rounded">"${esc(w.matched_text)}"</code>` : "";
    return `<div class="flex gap-2.5 px-4 py-2.5 hover:bg-white/50 dark:hover:bg-slate-800/40 transition-colors">
  <div class="shrink-0 mt-0.5 w-5 h-5 rounded-full ${c.ring} flex items-center justify-center">
    <svg class="w-3 h-3 ${c.txt}" fill="currentColor" viewBox="0 0 20 20">${ICONS[w.severity]||ICONS.info}</svg>
  </div>
  <div class="min-w-0 flex-1">
    <div class="flex items-center gap-1.5 flex-wrap mb-0.5">
      <span class="text-xs font-bold text-gray-800 dark:text-gray-100">${esc(w.title)}</span>
      <span class="text-xs ${c.txt} font-medium opacity-80">${esc(w.category)}</span>
    </div>
    <p class="text-xs text-gray-600 dark:text-gray-400 leading-relaxed">${esc(w.message)}</p>
    ${mb}
    <div class="mt-1 flex items-start gap-1"><svg class="w-3 h-3 text-green-500 mt-0.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M9 12l2 2 4-4"/></svg>
    <p class="text-xs text-green-600 dark:text-green-400 leading-relaxed">${esc(w.suggestion)}</p></div>
  </div>
</div>`;
  }

  // ── Batch panel HTML ─────────────────────────────────────────
  function buildBatchPanel(id) {
    return `<div id="${id}" class="ambi-batch-panel hidden rounded-2xl border border-gray-200 dark:border-slate-700 bg-white/70 dark:bg-slate-800/70 overflow-hidden shadow-sm">
  <div class="flex items-center justify-between px-5 py-4 bg-amber-50 dark:bg-amber-900/20 border-b border-gray-200 dark:border-slate-700">
    <div class="flex items-center gap-3">
      <div class="w-8 h-8 rounded-lg bg-amber-100 dark:bg-amber-900/40 flex items-center justify-center shrink-0">
        <svg class="w-4 h-4 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>
      </div>
      <div><h3 class="text-sm font-bold text-gray-900 dark:text-white">Ambiguity Pre-Scan Report</h3>
      <p class="text-xs text-gray-500 dark:text-gray-400">Review quality issues before classification</p></div>
    </div>
    <button onclick="document.getElementById('${id}').classList.add('hidden')" class="text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition">
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
    </button>
  </div>
  <div class="grid grid-cols-2 md:grid-cols-5 gap-3 px-5 py-4 bg-gray-50 dark:bg-slate-900/40 border-b border-gray-200 dark:border-slate-700">
    <div class="text-center"><p class="text-xl font-bold text-gray-900 dark:text-white ambi-bs-total">—</p><p class="text-xs text-gray-500 uppercase font-medium mt-0.5">Total</p></div>
    <div class="text-center"><p class="text-xl font-bold text-green-600 dark:text-green-400 ambi-bs-clean">—</p><p class="text-xs text-gray-500 uppercase font-medium mt-0.5">✅ Clean</p></div>
    <div class="text-center"><p class="text-xl font-bold text-amber-500 ambi-bs-amb">—</p><p class="text-xs text-gray-500 uppercase font-medium mt-0.5">⚠️ Ambiguous</p></div>
    <div class="text-center"><p class="text-xl font-bold text-blue-500 ambi-bs-avg">—</p><p class="text-xs text-gray-500 uppercase font-medium mt-0.5">Avg Score</p></div>
    <div class="text-center"><div class="flex flex-wrap justify-center gap-1 ambi-bs-mix">—</div><p class="text-xs text-gray-500 uppercase font-medium mt-0.5">Quality Mix</p></div>
  </div>
  <div class="px-5 py-3 flex items-center gap-2 flex-wrap border-b border-gray-100 dark:border-slate-700">
    <span class="text-xs text-gray-500 font-medium">Filter:</span>
    <button onclick="Ambiguity._bf(this,'all')"      data-bpid="${id}" data-f="all"      class="ambi-bfbtn active-bf text-xs px-2.5 py-1 rounded-full font-medium bg-gray-200 dark:bg-slate-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 transition ring-2 ring-offset-1 ring-gray-400">All</button>
    <button onclick="Ambiguity._bf(this,'Critical')" data-bpid="${id}" data-f="Critical" class="ambi-bfbtn text-xs px-2.5 py-1 rounded-full font-medium bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 hover:bg-red-200 transition">🚨 Critical</button>
    <button onclick="Ambiguity._bf(this,'Poor')"     data-bpid="${id}" data-f="Poor"     class="ambi-bfbtn text-xs px-2.5 py-1 rounded-full font-medium bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400 hover:bg-orange-200 transition">🔶 Poor</button>
    <button onclick="Ambiguity._bf(this,'Fair')"     data-bpid="${id}" data-f="Fair"     class="ambi-bfbtn text-xs px-2.5 py-1 rounded-full font-medium bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 hover:bg-amber-200 transition">⚠️ Fair</button>
    <button onclick="Ambiguity._bf(this,'Good')"     data-bpid="${id}" data-f="Good"     class="ambi-bfbtn text-xs px-2.5 py-1 rounded-full font-medium bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 hover:bg-green-200 transition">✅ Good</button>
    <button onclick="Ambiguity._exportCSV('${id}')" class="ml-auto text-xs px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold transition flex items-center gap-1">
      <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg> Export
    </button>
  </div>
  <div class="ambi-loader hidden py-8 flex flex-col items-center justify-center gap-2">
    <div class="w-7 h-7 border-4 border-amber-400 border-t-transparent rounded-full animate-spin"></div>
    <p class="text-xs text-gray-500 animate-pulse">Scanning requirements…</p>
  </div>
  <div class="overflow-x-auto">
    <table class="w-full text-sm text-left">
      <thead class="bg-gray-50 dark:bg-slate-800 text-xs uppercase font-semibold text-gray-500 dark:text-gray-400">
        <tr><th class="px-4 py-2.5 w-8">#</th><th class="px-4 py-2.5">Requirement</th><th class="px-4 py-2.5 w-24">Quality</th><th class="px-4 py-2.5 w-32">Score</th><th class="px-4 py-2.5">Issues</th></tr>
      </thead>
      <tbody class="ambi-tbody divide-y divide-gray-100 dark:divide-slate-700 text-gray-700 dark:text-gray-300"></tbody>
    </table>
  </div>
</div>`;
  }

  function renderBatchPanel(data, panelEl) {
    if (!panelEl) return;
    panelEl.classList.remove("hidden");
    const s = data.summary || {};
    panelEl.querySelector(".ambi-bs-total").textContent = s.total ?? "—";
    panelEl.querySelector(".ambi-bs-clean").textContent = s.clean ?? "—";
    panelEl.querySelector(".ambi-bs-amb").textContent   = s.ambiguous ?? "—";
    panelEl.querySelector(".ambi-bs-avg").textContent   = s.avg_score ?? "—";
    const pm = { Good:"bg-green-100 text-green-700",Fair:"bg-amber-100 text-amber-700",Poor:"bg-orange-100 text-orange-700",Critical:"bg-red-100 text-red-700" };
    panelEl.querySelector(".ambi-bs-mix").innerHTML = Object.entries(s.label_counts||{}).map(([k,v])=>
      `<span class="text-xs px-1 py-0.5 rounded font-medium ${pm[k]||""}">${k}:${v}</span>`).join("");
    panelEl._ambiResults = data.results || [];
    panelEl._ambiFilter  = "all";
    _renderRows(panelEl, "all");
    panelEl.scrollIntoView({ behavior:"smooth", block:"start" });
  }

  const BAR_MAP   = { Good:"bg-green-500",Fair:"bg-amber-400",Poor:"bg-orange-500",Critical:"bg-red-500" };
  const BADGE_MAP = { Good:"bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",Fair:"bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",Poor:"bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400",Critical:"bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400" };

  function _renderRows(panelEl, filter) {
    const results  = panelEl._ambiResults || [];
    const filtered = filter === "all" ? results : results.filter(r => r.quality_label === filter);
    const tbody    = panelEl.querySelector(".ambi-tbody");
    if (!filtered.length) {
      tbody.innerHTML = `<tr><td colspan="5" class="px-4 py-8 text-center text-gray-400 italic text-xs">No requirements match this filter.</td></tr>`;
      return;
    }
    tbody.innerHTML = filtered.map((r, i) => {
      const origIdx = results.indexOf(r) + 1;
      const pills   = (r.warnings||[]).slice(0,3).map(w=>`<span class="text-xs px-1 py-0.5 rounded font-medium ${SEVC[w.severity]?.pill||""}">${esc(w.title)}</span>`).join(" ");
      const detId   = `${panelEl.id}-dr-${i}`;
      const allW    = (r.warnings||[]).map(w=>`<div class="flex gap-2 py-1 border-t border-gray-100 dark:border-slate-700 first:border-t-0">
        <span class="text-xs px-1 py-0.5 rounded shrink-0 font-medium ${SEVC[w.severity]?.pill||""}">${w.severity}</span>
        <div><p class="text-xs font-semibold text-gray-700 dark:text-gray-300">${esc(w.title)}</p>
        <p class="text-xs text-gray-500 dark:text-gray-400">${esc(w.message)}</p>
        <p class="text-xs text-green-600 dark:text-green-400 mt-0.5">💡 ${esc(w.suggestion)}</p></div></div>`).join("");
      return `<tr class="hover:bg-gray-50 dark:hover:bg-slate-800/40 cursor-pointer transition-colors" onclick="document.getElementById('${detId}').classList.toggle('hidden')">
  <td class="px-4 py-2.5 text-xs text-gray-400 font-mono">${origIdx}</td>
  <td class="px-4 py-2.5 max-w-xs"><p class="text-xs truncate" title="${esc(r.requirement)}">${esc(r.requirement)}</p></td>
  <td class="px-4 py-2.5"><span class="text-xs px-2 py-0.5 rounded-full font-semibold ${BADGE_MAP[r.quality_label]||""}">${SC[r.quality_label]?.icon||""} ${r.quality_label}</span></td>
  <td class="px-4 py-2.5"><div class="flex items-center gap-2"><div class="w-14 bg-gray-200 dark:bg-slate-700 rounded-full h-1.5"><div class="h-1.5 rounded-full ${BAR_MAP[r.quality_label]||"bg-gray-400"}" style="width:${r.ambiguity_score}%"></div></div><span class="text-xs font-mono text-gray-500">${r.ambiguity_score}</span></div></td>
  <td class="px-4 py-2.5"><div class="flex flex-wrap gap-1">${pills||'<span class="text-xs text-gray-400 italic">None</span>'}</div></td>
</tr>
<tr id="${detId}" class="hidden"><td></td><td colspan="4" class="px-4 pb-3 pt-1 text-xs">
  <p class="font-medium text-gray-600 dark:text-gray-400 mb-1">Full text: <em class="text-gray-500">${esc(r.requirement)}</em></p>
  ${allW||'<p class="text-xs text-green-500">✅ No issues.</p>'}
</td></tr>`;
    }).join("");
  }

  function _bf(btn, filter) {
    const pid = btn.dataset.bpid;
    const p   = document.getElementById(pid);
    if (!p) return;
    p._ambiFilter = filter;
    p.querySelectorAll(".ambi-bfbtn").forEach(b => {
      const active = b.dataset.f === filter;
      b.classList.toggle("ring-2",        active);
      b.classList.toggle("ring-offset-1", active);
      b.classList.toggle("ring-gray-400", active);
    });
    _renderRows(p, filter);
  }

  function _exportCSV(panelId) {
    const p = document.getElementById(panelId);
    const results = p?._ambiResults || [];
    if (!results.length) return;
    const hdr = ["#","Requirement","Quality","Score","Issues"];
    const rows = results.map((r,i)=>[i+1,`"${(r.requirement||"").replace(/"/g,'""')}"`,r.quality_label,r.ambiguity_score,`"${(r.warnings||[]).map(w=>w.title).join("; ")}"`]);
    const csv  = [hdr,...rows].map(r=>r.join(",")).join("\n");
    const a    = Object.assign(document.createElement("a"),{href:URL.createObjectURL(new Blob([csv],{type:"text/csv"})),download:"ambiguity_report.csv"});
    a.click(); URL.revokeObjectURL(a.href);
  }

  function _toggle(id) {
    const p = document.getElementById(id); if (!p) return;
    const body = p.querySelector(".ambi-body");
    const chev = p.querySelector(".ambi-chev");
    const open = !body.classList.contains("hidden");
    body.classList.toggle("hidden", open);
    if (chev) chev.style.transform = open ? "rotate(-90deg)" : "rotate(0deg)";
  }

  // ── attach to a textarea ─────────────────────────────────────
  function attach(ta, insertAfter) {
    if (!ta || ta._ambiAttached) return;
    ta._ambiAttached = true;
    const id  = `ambi-p-${++_pid}`;
    const target = insertAfter || ta;
    const wrap = document.createElement("div");
    wrap.innerHTML = buildInlinePanel(id);
    target.insertAdjacentElement("afterend", wrap.firstElementChild);
    const panelEl = document.getElementById(id);
    const run = deb(async () => {
      const text = ta.value.trim();
      if (!text) { panelEl.classList.add("hidden"); return; }
      try { const r = await scanSingle(text); panelEl.classList.remove("hidden"); renderInline(r, panelEl); }
      catch(e) { console.warn("[Ambiguity]", e); }
    }, 550);
    ta.addEventListener("input", run);
    if (ta.value.trim()) run();
  }

  // ── auto-wire known textarea IDs ────────────────────────────
  const AUTO_IDS = ["storyInput","contextStoryInput","singleStory","compareStoryInput","annoStoryInput","reqInput"];
  function autoWire() {
    AUTO_IDS.forEach(id => { const el = document.getElementById(id); if (el) attach(el); });
    document.querySelectorAll("textarea[data-ambiguity]").forEach(el => attach(el));
  }

  // ── Batch helper: run scan + inject panel ────────────────────
  async function runBatchScan(stories, containerEl, panelId) {
    if (!containerEl) return;
    panelId = panelId || `ambi-batch-${++_pid}`;
    if (!document.getElementById(panelId)) {
      containerEl.innerHTML = buildBatchPanel(panelId);
    }
    const p = document.getElementById(panelId);
    p.classList.remove("hidden");
    p.querySelector(".ambi-loader").classList.remove("hidden");
    p.querySelector(".ambi-loader").classList.add("flex");
    p.querySelector(".ambi-tbody").innerHTML = "";
    try {
      const data = await scanBatch(stories);
      renderBatchPanel(data, p);
    } catch(e) {
      p.querySelector(".ambi-tbody").innerHTML = `<tr><td colspan="5" class="px-4 py-6 text-center text-red-500 text-xs">Scan failed: ${e.message}</td></tr>`;
    } finally {
      p.querySelector(".ambi-loader").classList.add("hidden");
      p.querySelector(".ambi-loader").classList.remove("flex");
    }
    return panelId;
  }

  // ── Public API ───────────────────────────────────────────────
  G.Ambiguity = {
    attach, autoWire, scanSingle, scanBatch,
    renderInline, renderBatchPanel, buildBatchPanel, buildInlinePanel,
    runBatchScan,
    _toggle, _bf, _exportCSV,
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", autoWire);
  else autoWire();

})(window);