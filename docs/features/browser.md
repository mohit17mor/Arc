# Browser Automation

Arc opens a **real Chromium browser** and interacts with pages like a human — using accessibility tree snapshots, not screenshots.

## How It Works

Each page is converted to a numbered list of interactive elements:

```
[3] textbox "Where from?" value="Delhi"
[5] combobox "Where to?"
[7] textbox "Departure" value=""
[9] button "Search"
```

The LLM sees this, decides what to do, and sends actions:

```json
{"actions": [
  {"type": "fill", "target": "[5]", "value": "Mumbai"},
  {"type": "fill", "target": "[7]", "value": "2026-04-10"},
  {"type": "click", "target": "[9]"}
]}
```

## Browser Tools

| Tool | Description |
|------|-------------|
| `browser_go` | Navigate to a URL, get a structured page snapshot |
| `browser_look` | Re-examine the current page |
| `browser_act` | Click, fill, scroll, submit — all in one call |

## Capabilities

- Fills forms (text, dropdowns, comboboxes, date pickers)
- Handles autocomplete suggestions (Google Flights, Amazon, etc.)
- Navigates calendars, picks dates, closes overlays
- Deals with CAPTCHAs by asking you to solve them, then continues
- Persistent browser profiles (cookies/sessions survive restarts)
- Switches between headless and headed mode for human assist

## Under the Hood

The engine handles complex interactions mechanically:

- **Autocomplete** — types char-by-char, waits for dropdown, picks best match using word-boundary-aware scoring
- **Calendars** — detects calendar type (data-iso, aria-label, gridcell), navigates months, clicks the right day
- **Overlays** — escalating click fallbacks (normal → force → JS → mouse coordinates)
- **CAPTCHAs** — detected and escalated to human, then continues where it left off

## Why Accessibility Tree?

| | Screenshot + Vision | Accessibility Tree |
|---|---|---|
| Speed | Slow (image encoding + vision model) | Fast (text only) |
| Cost | Expensive (vision tokens) | Cheap (text tokens) |
| Accuracy | Approximate coordinates | Exact element targeting |
| Works with | Vision-capable models only | Any LLM |
