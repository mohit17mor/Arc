# Liquid Web — Product Search & Comparison

Ask Arc to find products and it renders a **live 3D carousel** you can browse in your browser.

```
You:  "find me the best wireless earbuds under ₹5000"
Arc:  *searches → scrapes → renders → serves*
      "Found 12 products from 4 sites. Comparison page is live at: http://localhost:63350"
```

## Pipeline

1. **Search** — queries Tavily API to find relevant product pages
2. **Scrape** — launches parallel headless Chromium contexts via BrowserPool
3. **Extract** — pulls structured data using JSON-LD, OpenGraph, and DOM heuristics
4. **Filter** — scores products on quality and drops noise
5. **Deduplicate** — removes near-duplicate products by name similarity
6. **Render** — generates a responsive 3D carousel UI
7. **Serve** — local HTTP server (optionally tunneled via ngrok)

## Server Lifecycle

- Runs independently in the background
- Auto-shuts down after 10 minutes of inactivity
- A new search replaces the old one
- Cleans up when you exit `arc chat`

## Setup

```toml
[tavily]
api_key = "tvly-..."    # get a free key at tavily.com

[ngrok]
auth_token = "..."      # optional, for public URLs
```
