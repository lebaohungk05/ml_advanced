"use strict";

(function () {
  const STORAGE_KEY = "fashion_theme";
  const root = document.documentElement;

  function stored() {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch (err) {
      return null;
    }
  }

  function preferred() {
    const saved = stored();
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function apply(theme) {
    root.setAttribute("data-theme", theme);
    const toggle = document.getElementById("theme-toggle");
    if (!toggle) return;
    toggle.setAttribute("aria-pressed", String(theme === "dark"));
    if (window.I18N) window.I18N.refresh();
  }

  root.setAttribute("data-theme", preferred());

  document.addEventListener("DOMContentLoaded", () => {
    apply(root.getAttribute("data-theme"));
    document.getElementById("theme-toggle").addEventListener("click", () => {
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      apply(next);
      try {
        localStorage.setItem(STORAGE_KEY, next);
      } catch (err) {
        return;
      }
    });
  });
})();
