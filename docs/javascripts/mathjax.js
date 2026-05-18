window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
  }
};

function renderMathJax() {
  if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
    window.MathJax.typesetPromise();
  }
}

if (typeof document$ !== "undefined" && document$.subscribe) {
  document$.subscribe(renderMathJax);
} else {
  document.addEventListener("DOMContentLoaded", renderMathJax);
}

