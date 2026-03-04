"""
HTML renderer for Liquid Web product comparison UI.

Generates a responsive, mobile-first 3D carousel interface
from extracted product data.
"""

from __future__ import annotations

import json
import logging
from arc.liquid.extract import ProductData

logger = logging.getLogger(__name__)


def render_products(
    query: str,
    products: list[ProductData],
    *,
    sources: list[str] | None = None,
) -> str:
    """
    Render product data into a beautiful comparison UI.

    Args:
        query: Original search query.
        products: Extracted products to display.
        sources: List of source domains for the metadata line.

    Returns:
        Complete HTML page as a string.
    """
    n = len(products)
    product_dicts = []
    for p in products:
        d = p.to_dict()
        # Truncate long names for display
        if len(d.get("name", "")) > 90:
            d["name"] = d["name"][:87] + "..."
        product_dicts.append(d)

    products_json = json.dumps(product_dicts, ensure_ascii=False)
    sources_text = ", ".join(sources) if sources else "multiple sources"

    # Particles HTML
    particles = "".join(
        f'<div class="particle" style="left:{(i*17+7)%100}%;'
        f"animation-duration:{7+i%9}s;animation-delay:{i*0.8:.1f}s;"
        f'width:{1+i%3}px;height:{1+i%3}px;"></div>'
        for i in range(25)
    )

    return _TEMPLATE.format(
        query=_html_escape(query),
        n=n,
        sources_text=_html_escape(sources_text),
        products_json=products_json,
        particles=particles,
    )


def _html_escape(s: str) -> str:
    """Basic HTML escaping."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ── Template ──────────────────────────────────────────────────
# Uses {{ / }} for literal braces in CSS/JS (Python .format() escaping)

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Arc — {query}</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@200;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}

    :root {{
      --bg: #060918;
      --card-bg: rgba(10, 14, 32, 0.88);
      --text: #e8eaf6;
      --text-sec: #5c618a;
      --cyan: #22d3ee;
      --purple: #a855f7;
      --accent: #8b5cf6;
      --accent2: #06b6d4;
      --green: #34d399;
      --amber: #fbbf24;
      --glow-cyan: rgba(34,211,238,0.35);
      --glow-purple: rgba(168,85,247,0.35);
      --border-grad: linear-gradient(135deg, var(--cyan), var(--purple));
    }}

    html, body {{
      width:100%; height:100%;
      overflow: hidden;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: var(--bg);
      color: var(--text);
    }}

    /* ── Ambient background ── */
    .bg-layer {{
      position:fixed; inset:0; z-index:0; pointer-events:none;
      background:
        radial-gradient(ellipse 80% 50% at 25% 20%, rgba(168,85,247,0.10), transparent 70%),
        radial-gradient(ellipse 70% 40% at 75% 80%, rgba(34,211,238,0.07), transparent 70%),
        radial-gradient(ellipse 90% 90% at 50% 50%, rgba(6,9,24,0.95), transparent);
      animation: bg-drift 30s ease-in-out infinite alternate;
    }}
    @keyframes bg-drift {{
      0%   {{ opacity:1; filter: hue-rotate(0deg); }}
      100% {{ opacity:0.85; filter: hue-rotate(10deg); }}
    }}

    .particles {{ position:fixed; inset:0; pointer-events:none; z-index:0; overflow:hidden; }}
    .particle {{
      position:absolute; width:2px; height:2px;
      background: var(--cyan); border-radius:50%; opacity:0;
      animation: float-up linear infinite;
    }}
    @keyframes float-up {{
      0%   {{ opacity:0; transform:translateY(100vh) scale(0); }}
      10%  {{ opacity:0.4; }}
      90%  {{ opacity:0.15; }}
      100% {{ opacity:0; transform:translateY(-10vh) scale(1); }}
    }}

    /* ── Header ── */
    .header {{
      position:relative; z-index:10;
      text-align:center;
      padding: 24px 16px 10px;
      animation: fade-down 0.8s ease forwards;
      opacity:0; transform:translateY(-20px);
    }}
    @keyframes fade-down {{ to {{ opacity:1; transform:translateY(0); }} }}

    .logo {{
      display:inline-flex; align-items:center; gap:10px;
      font-size:9px; font-weight:800; letter-spacing:8px;
      text-transform:uppercase; margin-bottom:10px;
      background: var(--border-grad);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      background-clip:text;
    }}
    .logo-dot {{
      width:5px; height:5px; border-radius:50%;
      background: var(--cyan);
      box-shadow:0 0 10px var(--glow-cyan),0 0 25px var(--glow-cyan);
      animation: pulse-dot 2s ease-in-out infinite;
    }}
    @keyframes pulse-dot {{
      0%,100% {{ box-shadow:0 0 10px var(--glow-cyan),0 0 25px var(--glow-cyan); }}
      50%     {{ box-shadow:0 0 20px var(--glow-cyan),0 0 50px var(--glow-cyan); }}
    }}

    .header h1 {{
      font-size: clamp(18px, 3.5vw, 28px); font-weight:200; letter-spacing:-0.5px; margin-bottom:5px;
    }}
    .header h1 strong {{
      font-weight:700;
      background: var(--border-grad);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      background-clip:text;
    }}
    .meta {{ font-size:10px; color:var(--text-sec); letter-spacing:0.5px; }}
    .meta .sep {{
      display:inline-block; width:3px; height:3px;
      background:var(--text-sec); border-radius:50%;
      margin:0 8px; vertical-align:middle;
    }}

    /* ── Convex Arc Carousel ── */
    .carousel-wrap {{
      position:relative; z-index:5;
      width:100%; height: calc(100vh - 160px);
      display:flex; align-items:center; justify-content:center;
      perspective: 1400px;
      overflow: visible;
    }}

    .carousel {{
      position:relative;
      display:flex; align-items:center; justify-content:center;
      width:100%; height:100%;
      transform-style: preserve-3d;
    }}

    /* ── Product Card ── */
    .card {{
      position:absolute;
      width: 230px; height: 340px;
      border-radius: 16px;
      overflow: hidden;
      background: var(--card-bg);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      cursor: pointer;
      text-decoration: none;
      color: var(--text);
      display: flex;
      flex-direction: column;
      transition: all 0.6s cubic-bezier(0.23, 1, 0.32, 1);
      border: 1px solid transparent;
      opacity: 0;
      pointer-events: none;
      will-change: transform, opacity;
    }}

    /* Neon gradient border via pseudo-element */
    .card::before {{
      content:'';
      position:absolute; inset: -1px;
      border-radius: 17px;
      background: linear-gradient(135deg, var(--cyan), var(--purple), var(--cyan));
      background-size: 300% 300%;
      z-index: -1;
      opacity: 0.15;
      transition: opacity 0.5s ease;
    }}
    .card::after {{
      content:'';
      position:absolute; inset: 0;
      border-radius: 16px;
      background: var(--card-bg);
      z-index: -1;
    }}

    /* Visible cards */
    .card.visible {{
      opacity: 0.5;
      pointer-events: none;
    }}
    .card.visible::before {{
      opacity: 0.12;
      animation: none;
    }}

    /* Adjacent cards (distance 1) */
    .card.adjacent {{
      opacity: 0.7;
      pointer-events: auto;
    }}
    .card.adjacent::before {{
      opacity: 0.25;
    }}

    /* Active center card */
    .card.active {{
      opacity: 1;
      pointer-events: auto;
      box-shadow:
        0 0 40px rgba(34,211,238,0.12),
        0 0 80px rgba(168,85,247,0.08),
        0 25px 60px rgba(0,0,0,0.5);
    }}
    .card.active::before {{
      opacity: 0.6;
      animation: border-flow 4s linear infinite;
    }}
    @keyframes border-flow {{
      0%   {{ background-position: 0% 50%; }}
      50%  {{ background-position: 100% 50%; }}
      100% {{ background-position: 0% 50%; }}
    }}

    /* Glassmorphism edge highlight on active */
    .card.active::after {{
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.08),
        inset 1px 0 0 rgba(34,211,238,0.05),
        inset -1px 0 0 rgba(168,85,247,0.05);
    }}

    .card-img {{
      width:100%; height: 150px;
      background: rgba(255,255,255,0.93);
      display:flex; align-items:center; justify-content:center;
      padding: 12px; position:relative;
      border-radius: 16px 16px 0 0;
      overflow:hidden;
    }}
    .card-img::after {{
      content:''; position:absolute; bottom:0; left:0; right:0;
      height:24px;
      background:linear-gradient(to top, var(--card-bg), transparent);
    }}
    .card-img img {{
      max-width:78%; max-height:78%; object-fit:contain;
      filter: drop-shadow(0 2px 8px rgba(0,0,0,0.08));
      transition: transform 0.5s cubic-bezier(0.16,1,0.3,1);
    }}
    .card.active .card-img img {{ transform: scale(1.06); }}

    .card-body {{
      padding: 10px 14px 12px; flex:1;
      display:flex; flex-direction:column;
    }}
    .card-title {{
      font-size: 11px; font-weight: 500; line-height: 1.45;
      margin-bottom: 6px;
      display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
    }}

    .source-badge {{
      font-size: 8px; font-weight:600; letter-spacing:0.5px;
      color: var(--cyan); margin-bottom: 5px; text-transform:uppercase;
      opacity: 0.8;
    }}

    .rating {{ display:flex; align-items:center; gap:4px; margin-bottom:6px; }}
    .stars {{ display:flex; gap:1px; }}
    .star {{
      width:10px; height:10px; display:inline-block;
      background:rgba(255,255,255,0.08); border-radius:1px;
      clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);
    }}
    .star.filled {{ background: var(--amber); }}
    .rating-num {{ font-size:11px; font-weight:600; }}
    .review-count {{ font-size:8px; color:var(--text-sec); }}

    .card-price {{
      display:flex; align-items:baseline; gap:5px; flex-wrap:wrap; margin-bottom: 8px;
    }}
    .price {{
      font-size: 18px; font-weight:800;
      background:linear-gradient(135deg, #fff 0%, #c0c0e0 100%);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      background-clip:text;
    }}
    .original-price {{ font-size:10px; color:var(--text-sec); text-decoration:line-through; }}
    .discount {{
      font-size:8px; font-weight:700; color:var(--green);
      background:rgba(52,211,153,0.1); border:1px solid rgba(52,211,153,0.2);
      padding:1px 5px; border-radius:8px;
    }}

    .card-cta {{
      margin-top:auto; display:flex; align-items:center; justify-content:space-between;
      padding-top: 8px; border-top:1px solid rgba(255,255,255,0.05);
    }}
    .card-cta span {{
      font-size: 9px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase;
      background: var(--border-grad);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      background-clip:text;
    }}
    .card-cta svg {{ color:var(--cyan); transition:transform 0.3s; }}
    .card.active .card-cta svg {{ transform:translateX(3px); }}

    /* ── Navigation ── */
    .nav-btn {{
      position:absolute; z-index:20;
      top:50%; transform:translateY(-50%);
      width:44px; height:44px;
      border-radius:50%;
      border:1px solid rgba(34,211,238,0.2);
      background:rgba(10,14,32,0.7);
      backdrop-filter:blur(16px);
      color: var(--cyan);
      cursor:pointer;
      display:flex; align-items:center; justify-content:center;
      transition:all 0.3s ease;
      box-shadow:0 4px 20px rgba(0,0,0,0.3);
    }}
    .nav-btn:hover {{
      background:rgba(34,211,238,0.1);
      border-color:rgba(34,211,238,0.5);
      box-shadow:0 0 25px rgba(34,211,238,0.15);
      transform:translateY(-50%) scale(1.08);
    }}
    .nav-prev {{ left:clamp(8px, 2vw, 60px); }}
    .nav-next {{ right:clamp(8px, 2vw, 60px); }}

    .indicators {{
      position:absolute; bottom:14px; left:50%; transform:translateX(-50%);
      z-index:10; display:flex; gap:4px; align-items:center;
    }}
    .ind {{
      width:6px; height:6px; border-radius:50%;
      background:rgba(255,255,255,0.08);
      border:1px solid rgba(255,255,255,0.06);
      transition:all 0.4s ease;
      cursor:pointer;
    }}
    .ind.active {{
      background: var(--cyan);
      border-color: var(--cyan);
      box-shadow:0 0 8px var(--glow-cyan);
      transform:scale(1.3);
    }}

    .counter {{
      position:absolute; bottom:14px; right:clamp(8px,2vw,60px);
      z-index:10;
      font-size:10px; font-weight:600; color:var(--text-sec);
      letter-spacing:2px;
    }}
    .counter .current {{
      font-size:14px; font-weight:800;
      background: var(--border-grad);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      background-clip:text;
    }}

    .floor-glow {{
      position:absolute; bottom: -30px; left:50%; transform:translateX(-50%);
      width: 400px; height: 60px;
      background: radial-gradient(ellipse at center, rgba(34,211,238,0.06) 0%, rgba(168,85,247,0.03) 40%, transparent 70%);
      pointer-events:none;
      filter: blur(8px);
    }}

    .footer {{
      position:fixed; bottom:0; left:0; right:0;
      z-index:10; text-align:center;
      padding:8px 16px;
      font-size:8px; color:var(--text-sec); letter-spacing:0.5px;
      background:linear-gradient(to top, var(--bg), transparent);
    }}

    /* ── Mobile ── */
    @media (max-width: 700px) {{
      .card {{ width: 200px; height: 300px; }}
      .card-img {{ height: 120px; padding: 10px; }}
      .card-title {{ font-size: 10px; }}
      .price {{ font-size: 16px; }}
      .nav-btn {{ width: 36px; height: 36px; }}
      .header {{ padding: 16px 12px 6px; }}
    }}

    @media (max-width: 420px) {{
      .card {{ width: 170px; height: 270px; }}
      .card-img {{ height: 100px; padding: 8px; }}
      .card-body {{ padding: 8px 10px 10px; }}
      .card-title {{ font-size: 9px; -webkit-line-clamp: 2; }}
      .price {{ font-size: 14px; }}
      .source-badge {{ font-size: 7px; }}
      .nav-btn {{ width: 32px; height: 32px; }}
    }}
  </style>
</head>
<body>
  <div class="bg-layer"></div>
  <div class="particles">{particles}</div>

  <div class="header">
    <div class="logo"><div class="logo-dot"></div>ARC LIQUID WEB</div>
    <h1>Results for <strong>"{query}"</strong></h1>
    <p class="meta">{n} products found<span class="sep"></span>{sources_text}<span class="sep"></span>Swipe or use arrows</p>
  </div>

  <div class="carousel-wrap" id="carouselWrap">
    <button class="nav-btn nav-prev" id="prevBtn" aria-label="Previous">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 18l-6-6 6-6"/></svg>
    </button>

    <div class="carousel" id="carousel"></div>
    <div class="floor-glow"></div>

    <button class="nav-btn nav-next" id="nextBtn" aria-label="Next">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18l6-6-6-6"/></svg>
    </button>

    <div class="indicators" id="indicators"></div>
    <div class="counter"><span class="current" id="counterCur">1</span> / {n}</div>
  </div>

  <div class="footer">
    Rendered by Arc AI Agent &middot; Prices may vary &middot; Tap card to view
  </div>

  <script>
  (function() {{
    const products = {products_json};
    const N = products.length;
    if (N === 0) return;

    const carousel = document.getElementById('carousel');
    const indicators = document.getElementById('indicators');
    const counterCur = document.getElementById('counterCur');
    const wrap = document.getElementById('carouselWrap');
    let current = 0;

    // Create cards
    products.forEach((p, i) => {{
      const card = document.createElement('a');
      card.className = 'card';
      card.href = p.url || '#';
      card.target = '_blank';
      card.rel = 'noopener';

      const sourceBadge = p.source_domain
        ? '<div class="source-badge">' + p.source_domain + '</div>'
        : '';

      let priceHTML = '';
      if (p.price) {{
        const currency = p.currency || '';
        priceHTML = '<span class="price">' + currency + p.price + '</span>';
        if (p.original_price) {{
          const orig = parseFloat(String(p.original_price).replace(/,/g, ''));
          const cur = parseFloat(String(p.price).replace(/,/g, ''));
          if (orig > cur) {{
            const disc = Math.round(((orig - cur) / orig) * 100);
            priceHTML += '<span class="original-price">' + currency + p.original_price + '</span>';
            priceHTML += '<span class="discount">-' + disc + '%</span>';
          }}
        }}
      }}

      let ratingHTML = '';
      if (p.rating) {{
        const full = Math.floor(parseFloat(p.rating));
        let stars = '';
        for (let s = 0; s < 5; s++) {{
          stars += '<span class="star' + (s < full ? ' filled' : '') + '"></span>';
        }}
        const rev = p.review_count ? parseInt(p.review_count).toLocaleString() + ' reviews' : '';
        ratingHTML = '<div class="rating"><div class="stars">' + stars + '</div>'
          + '<span class="rating-num">' + p.rating + '</span>'
          + '<span class="review-count">' + rev + '</span></div>';
      }}

      const imgSrc = p.image_url || '';
      const imgTag = imgSrc
        ? '<img src="' + imgSrc + '" alt="" loading="lazy" onerror="this.style.display=\\x27none\\x27" />'
        : '<div style="color:#555;font-size:12px;">No image</div>';

      card.innerHTML =
        '<div class="card-img">' + imgTag + '</div>'
        + '<div class="card-body">'
        + sourceBadge
        + '<h3 class="card-title">' + (p.name || 'Unknown Product') + '</h3>'
        + ratingHTML
        + '<div class="card-price">' + priceHTML + '</div>'
        + '<div class="card-cta"><span>View Deal</span>'
        + '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>'
        + '</div></div>';

      carousel.appendChild(card);

      const dot = document.createElement('div');
      dot.className = 'ind' + (i === 0 ? ' active' : '');
      dot.onclick = () => goTo(i);
      indicators.appendChild(dot);
    }});

    // ── Convex arc layout ──
    // Cards are positioned along a convex arc that bows toward the viewer.
    // Center card: full size, facing forward, z pushed closest to viewer.
    // Side cards: rotated away, scaled down, pushed back in Z, spaced apart in X.
    function layoutArc() {{
      const cards = carousel.children;
      const wrapW = wrap.offsetWidth;

      // Responsive spacing
      const isMobile = wrapW < 500;
      const cardW = isMobile ? (wrapW < 420 ? 170 : 200) : 230;
      const xGap = isMobile ? cardW * 0.58 : cardW * 0.65;
      const maxVisible = isMobile ? 2 : 2;  // how many on each side of center

      for (let i = 0; i < N; i++) {{
        const offset = i - current;
        // Wrap offset for circular navigation
        let d = offset;
        if (d > N / 2) d -= N;
        if (d < -N / 2) d += N;

        const absD = Math.abs(d);
        const card = cards[i];

        // Remove all state classes
        card.classList.remove('active', 'adjacent', 'visible');

        if (absD > maxVisible + 1) {{
          // Far away — hide completely
          card.style.transform = 'translateX(0) translateZ(-600px) rotateY(0deg) scale(0.5)';
          card.style.opacity = '0';
          card.style.pointerEvents = 'none';
          card.style.zIndex = '0';
          continue;
        }}

        // Scale: center=1, each step shrinks
        const scale = Math.max(0.55, 1 - absD * 0.18);
        // X position: spread outward
        const tx = d * xGap;
        // Z position: center closest, sides pushed back (convex = bowing toward viewer)
        const tz = -(absD * absD) * (isMobile ? 80 : 120);
        // Y rotation: sides angle away from viewer
        const ry = d * (isMobile ? 20 : 28);

        card.style.transform = 'translateX(' + tx + 'px) translateZ(' + tz + 'px) rotateY(' + ry + 'deg) scale(' + scale + ')';
        card.style.zIndex = String(100 - absD * 10);

        if (absD === 0) {{
          card.classList.add('active');
          card.style.opacity = '1';
          card.style.pointerEvents = 'auto';
        }} else if (absD === 1) {{
          card.classList.add('adjacent');
          card.style.opacity = '0.7';
          card.style.pointerEvents = 'auto';
        }} else {{
          card.classList.add('visible');
          card.style.opacity = '0.35';
          card.style.pointerEvents = 'none';
        }}
      }}
    }}

    function goTo(idx) {{
      current = ((idx % N) + N) % N;
      layoutArc();

      const dots = indicators.children;
      for (let i = 0; i < N; i++) {{
        dots[i].classList.toggle('active', i === current);
      }}
      counterCur.textContent = current + 1;
    }}

    goTo(0);

    // Navigation
    document.getElementById('prevBtn').onclick = () => goTo(current - 1);
    document.getElementById('nextBtn').onclick = () => goTo(current + 1);

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') goTo(current - 1);
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') goTo(current + 1);
    }});

    // Touch swipe
    let touchStartX = 0;
    wrap.addEventListener('touchstart', (e) => {{
      touchStartX = e.changedTouches[0].screenX;
    }}, {{ passive: true }});
    wrap.addEventListener('touchend', (e) => {{
      const dx = e.changedTouches[0].screenX - touchStartX;
      if (Math.abs(dx) > 40) {{
        if (dx < 0) goTo(current + 1);
        else goTo(current - 1);
      }}
    }}, {{ passive: true }});

    // Mouse drag
    let dragStartX = 0, dragging = false;
    wrap.addEventListener('mousedown', (e) => {{
      dragStartX = e.clientX; dragging = true; e.preventDefault();
    }});
    window.addEventListener('mouseup', (e) => {{
      if (!dragging) return;
      dragging = false;
      const dx = e.clientX - dragStartX;
      if (Math.abs(dx) > 50) {{
        if (dx < 0) goTo(current + 1);
        else goTo(current - 1);
      }}
    }});

    // Scroll wheel
    let wheelCooldown = false;
    wrap.addEventListener('wheel', (e) => {{
      if (wheelCooldown) return;
      wheelCooldown = true;
      setTimeout(() => wheelCooldown = false, 400);
      if (e.deltaY > 0 || e.deltaX > 0) goTo(current + 1);
      else goTo(current - 1);
    }}, {{ passive: true }});

    // Auto-rotate (stops on first interaction)
    let autoTimer = setInterval(() => goTo(current + 1), 5000);
    ['mousedown','touchstart','keydown'].forEach(evt =>
      wrap.addEventListener(evt, () => {{ clearInterval(autoTimer); }}, {{ once: true }})
    );

    // Re-layout on resize
    window.addEventListener('resize', () => layoutArc());
  }})();
  </script>
</body>
</html>"""
