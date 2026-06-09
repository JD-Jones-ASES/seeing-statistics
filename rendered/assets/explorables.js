/* explorables.js — bespoke, dependency-free interactive statistics widgets.
   Each reads real data from window.SC_DATA, computes live in the browser, and
   redraws on every slider move. Colours match the notebooks (Okabe-Ito). */
(function () {
  "use strict";

  var OK = { capture: "#009E73", miss: "#D55E00", accent: "#4c72b0",
             amber: "#E69F00", neutral: "#9aa7c7", purple: "#CC79A7", black: "#222" };

  function cssVar(n, f) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(n).trim();
    return v || f;
  }
  var ink = function () { return cssVar("--sc-ink", "#1f2330"); };
  var muted = function () { return cssVar("--sc-muted", "#5b6577"); };
  var grid = function () { return cssVar("--sc-line", "#e3e7ef"); };
  var card = function () { return cssVar("--sc-card", "#ffffff"); };

  // ---------- maths ----------
  function mean(a) { var s = 0, i; for (i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function sd(a, ddof) { var m = mean(a), s = 0, i, d; for (i = 0; i < a.length; i++) { d = a[i] - m; s += d * d; } return Math.sqrt(s / (a.length - (ddof || 0))); }
  function erf(x) {
    var s = x < 0 ? -1 : 1; x = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * x);
    var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  function normCDF(z) { return 0.5 * (1 + erf(z / Math.SQRT2)); }
  function normPDF(x, mu, s) { var z = (x - mu) / s; return Math.exp(-0.5 * z * z) / (s * Math.sqrt(2 * Math.PI)); }

  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function sampleWR(pop, n, rng) { var out = new Array(n), L = pop.length, i; for (i = 0; i < n; i++) out[i] = pop[(rng() * L) | 0]; return out; }
  function sampleWOR(pop, n, rng) {
    var L = pop.length; if (n >= L) return pop.slice();
    var idx = new Array(L), i, j, t; for (i = 0; i < L; i++) idx[i] = i;
    for (i = 0; i < n; i++) { j = i + ((rng() * (L - i)) | 0); t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    var out = new Array(n); for (i = 0; i < n; i++) out[i] = pop[idx[i]]; return out;
  }
  function histogram(a, k, lo, hi) {
    var mn = lo, mx = hi, i, v, b;
    if (mn === undefined) { mn = Infinity; mx = -Infinity; for (i = 0; i < a.length; i++) { v = a[i]; if (v < mn) mn = v; if (v > mx) mx = v; } }
    if (mn === mx) mx = mn + 1;
    var w = (mx - mn) / k, counts = new Array(k); for (i = 0; i < k; i++) counts[i] = 0;
    for (i = 0; i < a.length; i++) { b = ((a[i] - mn) / w) | 0; if (b < 0) b = 0; if (b >= k) b = k - 1; counts[b]++; }
    var mc = 0; for (i = 0; i < k; i++) if (counts[i] > mc) mc = counts[i];
    return { mn: mn, mx: mx, w: w, counts: counts, max: mc };
  }
  function fmt(x) { return Math.abs(x) >= 1000 ? Math.round(x).toLocaleString() : (Math.round(x * 100) / 100).toString(); }

  // ---------- canvas ----------
  function fit(canvas, aspect) {
    var cssW = canvas.clientWidth || (canvas.parentElement && canvas.parentElement.clientWidth) || 820;
    var cssH = Math.round(cssW / aspect);
    var dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
    canvas.style.height = cssH + "px";
    var ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = card(); ctx.fillRect(0, 0, cssW, cssH);
    return { ctx: ctx, W: cssW, H: cssH };
  }
  function axes(ctx, m, xlab, x0, x1, ylab) {
    ctx.strokeStyle = grid(); ctx.lineWidth = 1; ctx.fillStyle = muted();
    ctx.font = "12px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.beginPath(); ctx.moveTo(m.l, m.b); ctx.lineTo(m.r, m.b); ctx.stroke();
    var ticks = 5, i, xv, px;
    for (i = 0; i <= ticks; i++) {
      xv = x0 + (x1 - x0) * i / ticks; px = m.l + (m.r - m.l) * i / ticks;
      ctx.strokeStyle = grid(); ctx.beginPath(); ctx.moveTo(px, m.b); ctx.lineTo(px, m.b + 4); ctx.stroke();
      ctx.fillText(fmt(xv), px, m.b + 7);
    }
    ctx.fillStyle = muted(); ctx.fillText(xlab, (m.l + m.r) / 2, m.b + 24);
  }

  // helper: make a labelled slider control
  function slider(parent, label, min, max, val, step, onInput, fmtFn) {
    var wrap = document.createElement("div"); wrap.className = "xp-ctl";
    var lab = document.createElement("label"); var span = document.createElement("span");
    span.className = "val"; span.textContent = fmtFn ? fmtFn(val) : val;
    lab.textContent = label + " "; lab.appendChild(span);
    var input = document.createElement("input"); input.type = "range";
    input.min = min; input.max = max; input.value = val; input.step = step || 1;
    input.addEventListener("input", function () {
      var v = parseFloat(input.value); span.textContent = fmtFn ? fmtFn(v) : v; onInput(v);
    });
    wrap.appendChild(lab); wrap.appendChild(input); parent.appendChild(wrap);
    return input;
  }
  function segmented(parent, label, opts, val, onPick) {
    var wrap = document.createElement("div"); wrap.className = "xp-ctl";
    var lab = document.createElement("label"); lab.textContent = label; wrap.appendChild(lab);
    var seg = document.createElement("div"); seg.className = "xp-seg";
    opts.forEach(function (o) {
      var b = document.createElement("button"); b.type = "button"; b.textContent = o.label;
      if (o.value === val) b.className = "on";
      b.addEventListener("click", function () {
        seg.querySelectorAll("button").forEach(function (x) { x.className = ""; });
        b.className = "on"; onPick(o.value);
      });
      seg.appendChild(b);
    });
    wrap.appendChild(seg); parent.appendChild(wrap); return seg;
  }
  function legend(parent, items) {
    var l = document.createElement("div"); l.className = "xp-legend";
    items.forEach(function (it) {
      var s = document.createElement("span");
      s.innerHTML = '<span class="xp-swatch" style="border-top-color:' + it.c +
        (it.dashed ? ';border-top-style:dashed' : '') + '"></span>' + it.t;
      l.appendChild(s);
    });
    parent.appendChild(l); return l;
  }

  // ================= 1) Confidence-interval coverage =================
  function buildCI(root) {
    var D = window.SC_DATA, popKeys = Object.keys(D.pops);
    var state = { pop: "ames_area", n: 50, level: "95", seed: 7 };
    var card_ = document.createElement("div"); card_.className = "xp";
    var ctrls = document.createElement("div"); ctrls.className = "xp-controls"; card_.appendChild(ctrls);

    var selWrap = document.createElement("div"); selWrap.className = "xp-ctl";
    var selLab = document.createElement("label"); selLab.textContent = "Population"; selWrap.appendChild(selLab);
    var sel = document.createElement("select");
    popKeys.forEach(function (k) { var o = document.createElement("option"); o.value = k; o.textContent = D.pops[k].name; sel.appendChild(o); });
    sel.value = state.pop; sel.addEventListener("change", function () { state.pop = sel.value; draw(); });
    selWrap.appendChild(sel); ctrls.appendChild(selWrap);

    slider(ctrls, "Sample size n =", 5, 250, state.n, 1, function (v) { state.n = v; draw(); });
    segmented(ctrls, "Confidence level", [
      { label: "80%", value: "80" }, { label: "90%", value: "90" },
      { label: "95%", value: "95" }, { label: "99%", value: "99" }], state.level,
      function (v) { state.level = v; draw(); });
    var btnWrap = document.createElement("div"); btnWrap.className = "xp-ctl";
    btnWrap.appendChild(document.createElement("label"));
    var btn = document.createElement("button"); btn.className = "xp-btn"; btn.textContent = "Draw 100 fresh samples";
    btn.addEventListener("click", function () { state.seed = (state.seed * 1664525 + 1013904223) >>> 0; draw(); });
    btnWrap.appendChild(btn); ctrls.appendChild(btnWrap);

    var canvas = document.createElement("canvas"); canvas.setAttribute("role", "img"); card_.appendChild(canvas);
    legend(card_, [{ c: OK.capture, t: "caught μ" }, { c: OK.miss, t: "missed", dashed: true }, { c: ink(), t: "true mean μ" }]);
    var read = document.createElement("p"); read.className = "xp-readout"; read.setAttribute("aria-live", "polite"); card_.appendChild(read);
    root.appendChild(card_);

    function draw() {
      var pop = D.pops[state.pop], vals = pop.values, mu = pop.mean, n = state.n;
      var dfIdx = Math.min(n - 1, D.tcrit_max_df) - 1;
      var tstar = D.tcrit[state.level][dfIdx];
      var rng = mulberry32(state.seed), rows = [], hits = 0, lo = Infinity, hi = -Infinity, i;
      for (i = 0; i < 100; i++) {
        var s = sampleWOR(vals, n, rng), xb = mean(s), sdv = sd(s, 1), se = sdv / Math.sqrt(n);
        var l = xb - tstar * se, h = xb + tstar * se, hit = l <= mu && mu <= h;
        if (hit) hits++; if (l < lo) lo = l; if (h > hi) hi = h;
        rows.push({ l: l, h: h, xb: xb, hit: hit });
      }
      var pad = (hi - lo) * 0.04 || 1; lo -= pad; hi += pad;
      var f = fit(canvas, 820 / 470), ctx = f.ctx, m = { l: 54, r: f.W - 14, t: 14, b: f.H - 34 };
      var X = function (v) { return m.l + (m.r - m.l) * (v - lo) / (hi - lo); };
      var Y = function (i) { return m.t + (m.b - m.t) * i / 99; };
      // true mean line
      ctx.strokeStyle = ink(); ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(X(mu), m.t); ctx.lineTo(X(mu), m.b); ctx.stroke(); ctx.setLineDash([]);
      for (i = 0; i < 100; i++) {
        var r = rows[i], y = Y(i);
        ctx.strokeStyle = r.hit ? OK.capture : OK.miss; ctx.lineWidth = 1.3;
        ctx.setLineDash(r.hit ? [] : [3, 3]);
        ctx.beginPath(); ctx.moveTo(X(r.l), y); ctx.lineTo(X(r.h), y); ctx.stroke();
        ctx.setLineDash([]);
        if (!r.hit) { ctx.fillStyle = OK.miss; ctx.fillRect(X(r.xb) - 2, y - 2, 4, 4); }
      }
      axes(ctx, m, pop.unit, lo, hi);
      ctx.fillStyle = ink(); ctx.font = "12px system-ui"; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
      ctx.fillText("interval 1–100", m.l, m.t - 2);
      var miss = 100 - hits;
      read.innerHTML = "<b>" + hits + "</b> of 100 intervals caught the true mean μ = <b>" + fmt(mu) + " " + pop.unit +
        "</b> (" + miss + " missed). The " + state.level + "% method should catch it about <b>" + state.level +
        "%</b> of the time — the misses are the expected price of sampling, not mistakes.";
      canvas.setAttribute("aria-label", hits + " of 100 " + state.level + "% confidence intervals for the mean of " +
        pop.name + " at sample size " + n + " captured the true mean; " + miss + " missed.");
    }
    CI_draws.push(draw); draw();
  }

  // ================= 2) Central Limit Theorem =================
  function makeSynthetic() {
    var r = mulberry32(20260609), N = 4000, uni = [], bim = [], exps = [], i, u;
    for (i = 0; i < N; i++) {
      uni.push(r() * 100);
      bim.push((r() < 0.5 ? 25 + 8 * gauss(r) : 70 + 10 * gauss(r)));
      u = r(); exps.push(-Math.log(1 - u) * 20);
    }
    return { uniform: uni, bimodal: bim, exponential: exps };
  }
  function gauss(r) { var u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  function buildCLT(root) {
    var D = window.SC_DATA, syn = makeSynthetic();
    var POPS = {
      ames_area: { name: "House sizes — right-skewed", vals: D.pops.ames_area.values, unit: "sq ft" },
      life_exp: { name: "Life expectancy — left-skewed", vals: D.pops.life_exp.values, unit: "years" },
      exponential: { name: "Exponential — very skewed", vals: syn.exponential, unit: "" },
      bimodal: { name: "Bimodal — two humps", vals: syn.bimodal, unit: "" },
      uniform: { name: "Uniform — flat", vals: syn.uniform, unit: "" }
    };
    var state = { pop: "ames_area", n: 10 };
    var card_ = document.createElement("div"); card_.className = "xp";
    var ctrls = document.createElement("div"); ctrls.className = "xp-controls"; card_.appendChild(ctrls);

    var selWrap = document.createElement("div"); selWrap.className = "xp-ctl";
    var selLab = document.createElement("label"); selLab.textContent = "Population shape"; selWrap.appendChild(selLab);
    var sel = document.createElement("select");
    Object.keys(POPS).forEach(function (k) { var o = document.createElement("option"); o.value = k; o.textContent = POPS[k].name; sel.appendChild(o); });
    sel.addEventListener("change", function () { state.pop = sel.value; draw(); });
    selWrap.appendChild(sel); ctrls.appendChild(selWrap);
    slider(ctrls, "Sample size n =", 1, 100, state.n, 1, function (v) { state.n = v; draw(); });

    var canvas = document.createElement("canvas"); canvas.setAttribute("role", "img"); card_.appendChild(canvas);
    legend(card_, [{ c: OK.neutral, t: "distribution of sample means" }, { c: OK.miss, t: "CLT normal curve" }, { c: ink(), t: "true mean μ" }]);
    var read = document.createElement("p"); read.className = "xp-readout"; read.setAttribute("aria-live", "polite"); card_.appendChild(read);
    root.appendChild(card_);

    function draw() {
      var pop = POPS[state.pop], vals = pop.vals, mu = mean(vals), sigma = sd(vals, 0), n = state.n;
      var rng = mulberry32(424242), NS = 3000, means = new Array(NS), i;
      for (i = 0; i < NS; i++) means[i] = mean(sampleWR(vals, n, rng));
      var seEmp = sd(means, 0), seCLT = sigma / Math.sqrt(n);
      // x-domain: a few SDs of the population mean around mu (so axis is stable-ish)
      var halfPop = 3.2 * sigma, lo = mu - halfPop, hi = mu + halfPop;
      var f = fit(canvas, 820 / 430), ctx = f.ctx, m = { l: 52, r: f.W - 14, t: 16, b: f.H - 34 };
      var h = histogram(means, 38, lo, hi);
      var X = function (v) { return m.l + (m.r - m.l) * (v - lo) / (hi - lo); };
      // normalize histogram to density so the normal curve overlays
      var area = 0; for (i = 0; i < h.counts.length; i++) area += h.counts[i] * h.w; // counts*width
      var ymax = 0, dens = h.counts.map(function (c) { var d = c / (NS * h.w); if (d > ymax) ymax = d; return d; });
      var curveMax = normPDF(mu, mu, Math.max(seCLT, 1e-9)); if (curveMax > ymax) ymax = curveMax;
      ymax *= 1.1;
      var Y = function (d) { return m.b - (m.b - m.t) * d / ymax; };
      // bars
      ctx.fillStyle = OK.neutral;
      for (i = 0; i < dens.length; i++) {
        var x0 = X(h.mn + i * h.w), x1 = X(h.mn + (i + 1) * h.w);
        ctx.fillRect(x0, Y(dens[i]), Math.max(1, x1 - x0 - 1), m.b - Y(dens[i]));
      }
      // CLT normal curve
      ctx.strokeStyle = OK.miss; ctx.lineWidth = 2.4; ctx.beginPath();
      for (i = 0; i <= 220; i++) { var xv = lo + (hi - lo) * i / 220; var px = X(xv), py = Y(normPDF(xv, mu, Math.max(seCLT, 1e-9))); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
      ctx.stroke();
      // true mean
      ctx.strokeStyle = ink(); ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(X(mu), m.t); ctx.lineTo(X(mu), m.b); ctx.stroke(); ctx.setLineDash([]);
      axes(ctx, m, "sample mean" + (pop.unit ? " (" + pop.unit + ")" : ""), lo, hi);
      ctx.fillStyle = ink(); ctx.font = "13px system-ui"; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
      ctx.fillText("3,000 sample means, n = " + n, m.l, m.t + 4);
      read.innerHTML = "Spread of the sample means: <b>" + fmt(seEmp) + "</b>" + (pop.unit ? " " + pop.unit : "") +
        "; the CLT predicts σ/√n = " + fmt(sigma) + "/√" + n + " = <b>" + fmt(seCLT) + "</b>. " +
        (n >= 30 ? "Even from this population, the averages are essentially normal." :
          "Raise n and watch the shape settle into a bell and the spread shrink.");
      canvas.setAttribute("aria-label", "Distribution of 3000 sample means of size " + n + " from " + pop.name +
        "; observed spread " + fmt(seEmp) + " versus CLT prediction " + fmt(seCLT) + ".");
    }
    CLT_draws.push(draw); draw();
  }

  // ================= 3) p-values, errors & power =================
  function buildPower(root) {
    var ZSTAR = { one: { "0.01": 2.3263, "0.05": 1.6449, "0.10": 1.2816 },
                  two: { "0.01": 2.5758, "0.05": 1.9600, "0.10": 1.6449 } };
    var state = { delta: 1.5, alpha: "0.05", sides: "one" };
    var card_ = document.createElement("div"); card_.className = "xp";
    var ctrls = document.createElement("div"); ctrls.className = "xp-controls"; card_.appendChild(ctrls);
    slider(ctrls, "True effect δ (in SEs) =", 0, 4, state.delta, 0.1, function (v) { state.delta = v; draw(); }, function (v) { return v.toFixed(1); });
    segmented(ctrls, "Significance α", [{ label: "0.01", value: "0.01" }, { label: "0.05", value: "0.05" }, { label: "0.10", value: "0.10" }], state.alpha, function (v) { state.alpha = v; draw(); });
    segmented(ctrls, "Test", [{ label: "one-sided", value: "one" }, { label: "two-sided", value: "two" }], state.sides, function (v) { state.sides = v; draw(); });

    var canvas = document.createElement("canvas"); canvas.setAttribute("role", "img"); card_.appendChild(canvas);
    legend(card_, [{ c: OK.accent, t: "null world (no effect)" }, { c: OK.amber, t: "true world (effect δ)" }, { c: OK.miss, t: "reject region (Type I = α)" }]);
    var read = document.createElement("p"); read.className = "xp-readout"; read.setAttribute("aria-live", "polite"); card_.appendChild(read);
    root.appendChild(card_);

    function draw() {
      var d = state.delta, a = parseFloat(state.alpha), zc = ZSTAR[state.sides][state.alpha];
      var power, typeI = a;
      if (state.sides === "one") power = 1 - normCDF(zc - d);
      else power = normCDF(-zc - d) + (1 - normCDF(zc - d));
      var lo = -4, hi = 4 + d, f = fit(canvas, 820 / 380), ctx = f.ctx, m = { l: 20, r: f.W - 14, t: 16, b: f.H - 34 };
      var ymax = normPDF(0, 0, 1) * 1.18;
      var X = function (v) { return m.l + (m.r - m.l) * (v - lo) / (hi - lo); };
      var Y = function (yv) { return m.b - (m.b - m.t) * yv / ymax; };
      function shade(muc, from, to, color, alpha) {
        ctx.fillStyle = color; ctx.globalAlpha = alpha; ctx.beginPath(); ctx.moveTo(X(from), m.b);
        var steps = 120, i, xv; for (i = 0; i <= steps; i++) { xv = from + (to - from) * i / steps; ctx.lineTo(X(xv), Y(normPDF(xv, muc, 1))); }
        ctx.lineTo(X(to), m.b); ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
      }
      function curve(muc, color) {
        ctx.strokeStyle = color; ctx.lineWidth = 2.4; ctx.beginPath();
        var i, xv; for (i = 0; i <= 240; i++) { xv = lo + (hi - lo) * i / 240; var px = X(xv), py = Y(normPDF(xv, muc, 1)); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
        ctx.stroke();
      }
      // power region under alt (reject & effect real) — light amber
      if (state.sides === "one") shade(d, zc, hi, OK.capture, 0.20);
      else { shade(d, zc, hi, OK.capture, 0.20); shade(d, lo, -zc, OK.capture, 0.20); }
      // Type I region under null — vermillion
      if (state.sides === "one") shade(0, zc, hi, OK.miss, 0.28);
      else { shade(0, zc, hi, OK.miss, 0.28); shade(0, lo, -zc, OK.miss, 0.28); }
      curve(0, OK.accent); curve(d, OK.amber);
      // critical line(s)
      ctx.strokeStyle = muted(); ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(X(zc), m.t); ctx.lineTo(X(zc), m.b); ctx.stroke();
      if (state.sides === "two") { ctx.beginPath(); ctx.moveTo(X(-zc), m.t); ctx.lineTo(X(-zc), m.b); ctx.stroke(); }
      ctx.setLineDash([]);
      axes(ctx, m, "test statistic (standard errors)", lo, hi);
      read.innerHTML = "α (Type I — a false alarm) = <b>" + a.toFixed(2) + "</b> · " +
        "power (catching a real effect) = <b>" + (power * 100).toFixed(0) + "%</b> · " +
        "β (Type II — a miss) = <b>" + ((1 - power) * 100).toFixed(0) + "%</b>. " +
        (d === 0 ? "With no real effect, the reject rate is exactly α." :
          "Bigger effects (or larger samples → larger δ) are easier to detect; stricter α lowers power.");
      canvas.setAttribute("aria-label", "Null and alternative distributions; with effect " + d.toFixed(1) +
        " SEs and alpha " + a + " (" + state.sides + "-sided), power is " + (power * 100).toFixed(0) + " percent.");
    }
    POW_draws.push(draw); draw();
  }

  // ---------- init + responsive/theme redraw ----------
  var CI_draws = [], CLT_draws = [], POW_draws = [];
  function redrawAll() { [].concat(CI_draws, CLT_draws, POW_draws).forEach(function (d) { try { d(); } catch (e) {} }); }

  document.addEventListener("DOMContentLoaded", function () {
    var el = document.getElementById("explorable");
    if (!el || !window.SC_DATA) return;
    var kind = el.getAttribute("data-kind");
    if (kind === "ci") buildCI(el);
    else if (kind === "clt") buildCLT(el);
    else if (kind === "power") buildPower(el);
    var t = null;
    window.addEventListener("resize", function () { clearTimeout(t); t = setTimeout(redrawAll, 150); });
    new MutationObserver(redrawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  });
})();
