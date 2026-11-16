// Ask a question, show the answer and its quoted passages.
// Everything from the server is inserted as text (textContent), never as HTML.
(function () {
  "use strict";

  const config = window.DELPHI_CONFIG || {};
  const form = document.getElementById("ask");
  const question = document.getElementById("question");
  const counter = document.getElementById("counter");
  const submit = document.getElementById("submit");
  const result = document.getElementById("result");
  const badges = document.getElementById("badges");
  const answer = document.getElementById("answer");
  const declined = document.getElementById("declined");
  const sourcesHeading = document.getElementById("sources-heading");
  const citations = document.getElementById("citations");
  const error = document.getElementById("error");
  const MAX = 300;

  let widgetId = null;
  let token = "";

  function renderTurnstile() {
    if (!window.turnstile || widgetId !== null) return;
    widgetId = window.turnstile.render("#turnstile", {
      sitekey: config.turnstileSiteKey,
      callback: (t) => { token = t; },
      "expired-callback": () => { token = ""; },
      "error-callback": () => { token = ""; },
    });
  }
  // The Turnstile script loads with defer; render once it's there.
  window.addEventListener("load", renderTurnstile);

  function resetTurnstile() {
    token = "";
    if (window.turnstile && widgetId !== null) window.turnstile.reset(widgetId);
  }

  function updateCounter() {
    counter.textContent = `${question.value.length} / ${MAX}`;
  }
  question.addEventListener("input", updateCounter);

  document.querySelectorAll("button.example").forEach((b) => {
    b.addEventListener("click", () => {
      question.value = b.textContent;
      updateCounter();
      question.focus();
    });
  });

  function showError(message) {
    error.textContent = message;
    error.hidden = false;
  }

  function badge(text, kind) {
    const span = document.createElement("span");
    span.className = `badge ${kind}`;
    span.textContent = text;
    badges.appendChild(span);
  }

  function show(data) {
    badges.replaceChildren();
    citations.replaceChildren();
    declined.hidden = true;
    sourcesHeading.hidden = true;

    if (data.cache_only) badge("Daily budget reached: cached answers only", "cap");

    if (data.status === "over_cap") {
      answer.textContent =
        "Today's budget for new answers is used up, and this question hasn't been asked " +
        "before. Try one of the examples, or come back tomorrow (UTC).";
      result.hidden = false;
      return;
    }

    if (data.cache && data.cache.served_from_cache) {
      const sim = typeof data.cache.similarity === "number" ? data.cache.similarity.toFixed(3) : "";
      badge(`Served from cache${sim ? ` (${sim} match)` : ""}`, "cache");
    } else {
      badge("Fresh answer", "fresh");
    }

    answer.textContent = data.answer || "";
    declined.hidden = data.answerable !== false;

    if (data.citations && data.citations.length) {
      sourcesHeading.hidden = false;
      for (const c of data.citations) {
        const li = document.createElement("li");
        const quote = document.createElement("blockquote");
        quote.textContent = `“${c.quote}”`;
        const where = document.createElement("div");
        where.className = "where";
        where.textContent = `Life of ${c.life}, section ${c.section}`;
        li.append(quote, where);
        citations.appendChild(li);
      }
    }
    result.hidden = false;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    error.hidden = true;
    const text = question.value.trim();
    if (!text) return showError("Type a question first.");
    if (text.length > MAX) return showError(`Questions can be up to ${MAX} characters.`);
    if (!token) return showError("Please wait for the check below the examples to finish, then ask again.");

    submit.disabled = true;
    submit.textContent = "Asking…";
    result.setAttribute("aria-busy", "true");
    try {
      const response = await fetch(config.workerUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, turnstileToken: token }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        showError(data.error || "Something went wrong. Try again shortly.");
      } else {
        show(data);
      }
    } catch {
      showError("Couldn't reach Delphi. Check your connection and try again.");
    } finally {
      submit.disabled = false;
      submit.textContent = "Ask";
      result.setAttribute("aria-busy", "false");
      resetTurnstile(); // tokens are single-use
    }
  });

  updateCounter();
})();
