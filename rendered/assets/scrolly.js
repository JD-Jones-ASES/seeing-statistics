/* scrolly.js — a guided, scroll-driven telling of the flagship 100-CI story.
   A sticky canvas redraws as each prose step scrolls into view. Dependency-free
   (IntersectionObserver), theme-aware, reduced-motion friendly, over the real
   Ames population embedded in explorables-data.js. */
(function () {
  "use strict";
  var OK = { capture: "#009E73", miss: "#D55E00", accent: "#4c72b0", neutral: "#9aa7c7" };
  function cv(n, f) { var v = getComputedStyle(document.documentElement).getPropertyValue(n).trim(); return v || f; }
  function ink() { return cv("--sc-ink", "#1f2330"); }
  function muted() { return cv("--sc-muted", "#5b6577"); }
  function card() { return cv("--sc-card", "#fff"); }
  function mean(a) { for (var s = 0, i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function sd(a, d) { var m = mean(a), s = 0, i, x; for (i = 0; i < a.length; i++) { x = a[i] - m; s += x * x; } return Math.sqrt(s / (a.length - (d || 0))); }
  function mul(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function sampleWOR(pop, n, rng) { var L = pop.length, idx = new Array(L), i, j, t, o; for (i = 0; i < L; i++) idx[i] = i; for (i = 0; i < n; i++) { j = i + ((rng() * (L - i)) | 0); t = idx[i]; idx[i] = idx[j]; idx[j] = t; } o = new Array(n); for (i = 0; i < n; i++) o[i] = pop[idx[i]]; return o; }
  function fmt(x) { return Math.abs(x) >= 1000 ? Math.round(x).toLocaleString() : (Math.round(x * 10) / 10).toString(); }
  function fit(c, asp) {
    var W = c.clientWidth || 760, H = Math.round(W / asp), dpr = window.devicePixelRatio || 1;
    c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); c.style.height = H + "px";
    var ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = card(); ctx.fillRect(0, 0, W, H); return { ctx: ctx, W: W, H: H };
  }

  document.addEventListener("DOMContentLoaded", function () {
    var canvas = document.getElementById("scrolly-canvas");
    if (!canvas || !window.SC_DATA) return;
    var pop = SC_DATA.pops.ames_area.values, mu = SC_DATA.pops.ames_area.mean;
    var n = 50, tstar = SC_DATA.tcrit["95"][n - 2];
    // precompute 100 intervals (fixed seed) + one highlighted first sample
    var rng = mul(7), rows = [], i, hits = 0, lo = Infinity, hi = -Infinity, firstSample = null;
    for (i = 0; i < 100; i++) {
      var s = sampleWOR(pop, n, rng); if (i === 0) firstSample = s;
      var xb = mean(s), se = sd(s, 1) / Math.sqrt(n), l = xb - tstar * se, h = xb + tstar * se, hit = l <= mu && mu <= h;
      if (hit) hits++; if (l < lo) lo = l; if (h > hi) hi = h;
      rows.push({ l: l, h: h, xb: xb, hit: hit });
    }
    var pad = (hi - lo) * 0.05; lo -= pad; hi += pad;

    function popHist(ctx, m, W, H, fade) {
      var k = 44, mn = Math.min.apply(null, pop), mx = Math.max.apply(null, pop), w = (mx - mn) / k, counts = new Array(k).fill(0), j;
      for (j = 0; j < pop.length; j++) { var b = ((pop[j] - mn) / w) | 0; if (b < 0) b = 0; if (b >= k) b = k - 1; counts[b]++; }
      var cmax = Math.max.apply(null, counts);
      var X = function (v) { return m.l + (m.r - m.l) * (v - mn) / (mx - mn); };
      ctx.globalAlpha = fade ? 0.25 : 1; ctx.fillStyle = OK.neutral;
      for (j = 0; j < k; j++) { var x0 = X(mn + j * w), x1 = X(mn + (j + 1) * w), bh = (m.b - m.t) * counts[j] / cmax; ctx.fillRect(x0, m.b - bh, Math.max(1, x1 - x0 - 1), bh); }
      ctx.globalAlpha = 1; return X;
    }
    function muLine(ctx, m, X) {
      ctx.strokeStyle = ink(); ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(X(mu), m.t); ctx.lineTo(X(mu), m.b); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = ink(); ctx.font = "12px system-ui"; ctx.textAlign = "center";
      ctx.fillText("true mean μ = " + fmt(mu) + " sq ft", X(mu), m.t - 4);
    }
    function caterpillar(ctx, m, colored, emphasizeMiss) {
      var X = function (v) { return m.l + (m.r - m.l) * (v - lo) / (hi - lo); }, Y = function (i) { return m.t + (m.b - m.t) * i / 99; };
      for (var i = 0; i < 100; i++) {
        var r = rows[i], y = Y(i), isMiss = colored && !r.hit;
        ctx.strokeStyle = !colored ? OK.accent : (r.hit ? OK.capture : OK.miss);
        ctx.lineWidth = emphasizeMiss && isMiss ? 2.6 : 1.2; ctx.setLineDash(isMiss ? [3, 3] : []);
        ctx.beginPath(); ctx.moveTo(X(r.l), y); ctx.lineTo(X(r.h), y); ctx.stroke(); ctx.setLineDash([]);
      }
      muLine(ctx, m, X);
    }

    function draw(step) {
      var f = fit(canvas, 760 / 460), ctx = f.ctx, m = { l: 48, r: f.W - 16, t: 26, b: f.H - 22 };
      ctx.fillStyle = muted(); ctx.font = "12px system-ui"; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
      if (step <= 1) {
        var X = popHist(ctx, m, f.W, f.H, step === 1);
        muLine(ctx, m, X);
        ctx.fillStyle = muted(); ctx.fillText(step === 0 ? "2,930 homes (the whole population)" : "one sample of 50 (dots) → one interval", m.l, m.t - 12);
        if (step === 1 && firstSample) {
          for (var j = 0; j < firstSample.length; j++) { var px = X(firstSample[j]); ctx.strokeStyle = OK.accent; ctx.globalAlpha = 0.7; ctx.beginPath(); ctx.moveTo(px, m.b); ctx.lineTo(px, m.b - 12); ctx.stroke(); }
          ctx.globalAlpha = 1;
          var r0 = rows[0], yb = m.t + 18;
          ctx.strokeStyle = r0.hit ? OK.capture : OK.miss; ctx.lineWidth = 3;
          ctx.beginPath(); ctx.moveTo(X(r0.l), yb); ctx.lineTo(X(r0.h), yb); ctx.stroke();
          ctx.fillStyle = r0.hit ? OK.capture : OK.miss; ctx.fillText(r0.hit ? "this interval caught μ ✓" : "this interval missed ✕", X(r0.l), yb - 8);
        }
      } else {
        caterpillar(ctx, m, step >= 3, step >= 4);
        ctx.fillStyle = muted();
        ctx.fillText(step === 2 ? "100 samples → 100 intervals" :
          (step === 3 ? hits + " of 100 caught μ, " + (100 - hits) + " missed" :
            "the " + (100 - hits) + " misses (dashed) are the ~5% the method is allowed"), m.l, m.t - 12);
      }
    }

    var current = -1;
    function set(step) { if (step !== current) { current = step; draw(step); } }
    draw(0);
    var steps = document.querySelectorAll(".scrolly-step");
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) set(parseInt(e.target.getAttribute("data-step"), 10) || 0); });
    }, { rootMargin: "-45% 0px -45% 0px", threshold: 0 });
    steps.forEach(function (s) { io.observe(s); });
    var t; window.addEventListener("resize", function () { clearTimeout(t); t = setTimeout(function () { draw(current < 0 ? 0 : current); }, 150); });
    new MutationObserver(function () { draw(current < 0 ? 0 : current); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  });
})();
