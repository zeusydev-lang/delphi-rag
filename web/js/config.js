// Filled in at deploy time (docs/OPERATIONS.md, Website).
// Both values are public: the Worker URL is where questions go, and a
// Turnstile site key is meant to be embedded in pages.
window.DELPHI_CONFIG = {
  workerUrl: "https://delphi-api.REPLACE.workers.dev/api/ask",
  turnstileSiteKey: "REPLACE_WITH_TURNSTILE_SITE_KEY",
};
