/* course.js — shared behaviour for every rendered lesson page. Dependency-free.
   Handles: self-hosted KaTeX math rendering, copy-code buttons, a dark-mode
   toggle (saved per device), and a per-lesson "mark complete" progress tick
   (localStorage; works offline; never leaves the browser). */
(function () {
  "use strict";
  var LS_THEME = "sc-theme";
  var LS_DONE = "sc-progress";           // JSON map { lessonId: true }

  // ---- theme (light / dark), remembered on this device --------------------
  function applyTheme(t) {
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
  }
  try { applyTheme(localStorage.getItem(LS_THEME)); } catch (e) {}

  function currentTheme() {
    var explicit = document.documentElement.getAttribute("data-theme");
    if (explicit) return explicit;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  window.scToggleTheme = function () {
    var next = currentTheme() === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem(LS_THEME, next); } catch (e) {}
  };

  // ---- progress (per lesson, saved on this device) ------------------------
  function lessonId() {
    var m = document.querySelector('meta[name="sc-lesson"]');
    return m ? m.getAttribute("content") : null;
  }
  function readDone() {
    try { return JSON.parse(localStorage.getItem(LS_DONE) || "{}"); } catch (e) { return {}; }
  }
  function writeDone(map) {
    try { localStorage.setItem(LS_DONE, JSON.stringify(map)); } catch (e) {}
  }
  window.scProgress = { read: readDone, write: writeDone };

  document.addEventListener("DOMContentLoaded", function () {
    // KaTeX math (auto-render is loaded just before this script)
    if (window.renderMathInElement) {
      try {
        window.renderMathInElement(document.body, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "\\[", right: "\\]", display: true },
            { left: "\\(", right: "\\)", display: false },
            { left: "$", right: "$", display: false }
          ],
          throwOnError: false,
          ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
        });
      } catch (e) {}
    }

    // copy-code buttons on each code input
    document.querySelectorAll(".jp-CodeCell .jp-InputArea").forEach(function (area) {
      var src = area.querySelector(".jp-Editor, .highlight, pre");
      if (!src) return;
      var btn = document.createElement("button");
      btn.className = "sc-copy"; btn.type = "button"; btn.textContent = "Copy";
      btn.addEventListener("click", function () {
        var text = (area.querySelector(".highlight") || src).innerText.replace(/\n$/, "");
        navigator.clipboard.writeText(text).then(function () {
          btn.textContent = "Copied"; btn.classList.add("ok");
          setTimeout(function () { btn.textContent = "Copy"; btn.classList.remove("ok"); }, 1400);
        });
      });
      area.appendChild(btn);
    });

    // per-lesson "mark complete" checkbox (rendered in the footer)
    var box = document.getElementById("sc-done-box");
    var id = lessonId();
    if (box && id) {
      var done = readDone();
      box.checked = !!done[id];
      box.addEventListener("change", function () {
        var d = readDone();
        if (box.checked) d[id] = true; else delete d[id];
        writeDone(d);
      });
    }
  });
})();
