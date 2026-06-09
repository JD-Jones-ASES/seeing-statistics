/* quiz.js — render a lesson's self-check quiz from window.SC_QUIZ into #sc-quiz.
   Multiple-choice gives instant feedback; numeric questions check within a
   tolerance. Dependency-free; no answers leave the browser. */
(function () {
  "use strict";
  function el(tag, cls, txt) { var e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; }

  function renderQ(q, qi) {
    var box = el("div", "sc-q");
    box.appendChild(el("div", "stem", (qi + 1) + ". " + q.stem));
    var fb = el("div", "fb");
    if (q.type === "mc" && q.options) {
      q.options.forEach(function (opt, oi) {
        var o = el("div", "sc-opt", opt);
        o.setAttribute("role", "button"); o.setAttribute("tabindex", "0");
        function choose() {
          if (box.dataset.done) return; box.dataset.done = "1";
          var correct = oi === q.answer_index;
          o.classList.add(correct ? "correct" : "wrong");
          if (!correct) { var opts = box.querySelectorAll(".sc-opt"); if (opts[q.answer_index]) opts[q.answer_index].classList.add("correct"); }
          fb.className = "fb " + (correct ? "correct" : "wrong");
          fb.textContent = (correct ? "Correct. " : "Not quite. ") + (q.explain || "");
        }
        o.addEventListener("click", choose);
        o.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(); } });
        box.appendChild(o);
      });
    } else {
      var row = el("div", "num");
      var inp = el("input"); inp.type = "number"; inp.step = "any";
      inp.setAttribute("aria-label", "your numeric answer");
      inp.placeholder = "your answer" + (q.unit ? " (" + q.unit + ")" : "");
      var btn = el("button", "sc-btn", "Check"); btn.type = "button";
      function check() {
        var v = parseFloat(inp.value);
        if (isNaN(v)) { fb.className = "fb"; fb.textContent = "Enter a number."; return; }
        var tol = q.tolerance != null ? q.tolerance : Math.max(1e-9, Math.abs(q.answer) * 0.02);
        var ok = Math.abs(v - q.answer) <= tol;
        fb.className = "fb " + (ok ? "correct" : "wrong");
        fb.textContent = (ok ? "Correct. " : "The answer is " + q.answer + (q.unit ? " " + q.unit : "") + ". ") + (q.explain || "");
      }
      btn.addEventListener("click", check);
      inp.addEventListener("keydown", function (e) { if (e.key === "Enter") { e.preventDefault(); check(); } });
      row.appendChild(inp); row.appendChild(btn); box.appendChild(row);
    }
    box.appendChild(fb);
    return box;
  }

  document.addEventListener("DOMContentLoaded", function () {
    var mount = document.getElementById("sc-quiz");
    if (!mount || !window.SC_QUIZ) return;
    var id = mount.getAttribute("data-lesson");
    var qs = window.SC_QUIZ[id];
    if (!qs || !qs.length) return;
    var wrap = el("div", "sc-quiz");
    wrap.appendChild(el("h4", null, "Check yourself"));
    wrap.appendChild(el("p", null, "A few questions on this lesson — answers are checked in your browser.")).style.cssText = "color:var(--sc-muted);font-size:13.5px;margin:.1em 0 .5em";
    qs.forEach(function (q, qi) { wrap.appendChild(renderQ(q, qi)); });
    mount.appendChild(wrap);
  });
})();
