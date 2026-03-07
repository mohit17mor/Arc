"""
Local HTTP server + ngrok tunnel for Liquid Web.

Serves rendered HTML on a local port and exposes it via ngrok
so it's accessible from anywhere (Telegram, mobile, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import socket
import webbrowser

from aiohttp import web

logger = logging.getLogger(__name__)


class LiquidServer:
    """
    Serves Liquid Web HTML and exposes it via ngrok.

    Usage:
        server = LiquidServer(ngrok_auth_token="...")
        public_url = await server.start(html)
        # ... later ...
        await server.stop()
    """

    def __init__(self, ngrok_auth_token: str = "", auto_open: bool = True,
                 shutdown_timeout: float = 600.0):
        self._ngrok_token = ngrok_auth_token
        self._auto_open = auto_open
        self._shutdown_timeout = shutdown_timeout
        self._runner: web.AppRunner | None = None
        self._ngrok_tunnel = None
        self._port: int = 0
        self._public_url: str = ""
        self._shutdown_task: asyncio.Task | None = None

    async def start(self, html: str) -> str:
        """
        Start serving the HTML page and create ngrok tunnel.

        Args:
            html: The HTML content to serve.

        Returns:
            Public URL (ngrok if configured, otherwise localhost).
        """
        # Set up aiohttp app
        async def handle_index(request):
            return web.Response(text=html, content_type="text/html")

        app = web.Application()
        app.router.add_get("/", handle_index)

        self._runner = web.AppRunner(app)
        await self._runner.setup()

        # Find a free port
        self._port = _find_free_port()
        site = web.TCPSite(self._runner, "127.0.0.1", self._port)
        await site.start()
        logger.info("Liquid Web server started on port %d", self._port)

        local_url = f"http://localhost:{self._port}"

        # Create ngrok tunnel
        if self._ngrok_token:
            try:
                self._public_url = await self._create_tunnel()
                logger.info("Ngrok tunnel: %s → localhost:%d", self._public_url, self._port)
            except Exception as e:
                logger.warning("Ngrok tunnel failed, using local URL: %s", e)
                self._public_url = local_url
        else:
            self._public_url = local_url
            logger.info("No ngrok token configured, using local URL")

        # Auto-open in browser
        if self._auto_open:
            webbrowser.open(self._public_url)

        # Schedule auto-shutdown
        self._shutdown_task = asyncio.create_task(self._auto_shutdown())

        return self._public_url

    async def _create_tunnel(self) -> str:
        """Create ngrok tunnel and return public URL."""
        from pyngrok import ngrok, conf

        # Configure ngrok (runs in thread to avoid blocking)
        loop = asyncio.get_running_loop()

        def _setup_tunnel():
            conf.get_default().auth_token = self._ngrok_token
            tunnel = ngrok.connect(self._port, "http")
            return tunnel

        self._ngrok_tunnel = await loop.run_in_executor(None, _setup_tunnel)
        return self._ngrok_tunnel.public_url

    async def _auto_shutdown(self):
        """Auto-shutdown after timeout."""
        await asyncio.sleep(self._shutdown_timeout)
        logger.info("Liquid Web server auto-shutting down after %.0fs", self._shutdown_timeout)
        await self.stop()

    async def stop(self):
        """Stop server and close tunnel."""
        if self._shutdown_task and not self._shutdown_task.done():
            self._shutdown_task.cancel()

        if self._ngrok_tunnel:
            try:
                from pyngrok import ngrok
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, ngrok.disconnect, self._ngrok_tunnel.public_url)
            except Exception as e:
                logger.debug("Ngrok disconnect error: %s", e)
            self._ngrok_tunnel = None

        if self._runner:
            await self._runner.cleanup()
            self._runner = None

        logger.info("Liquid Web server stopped")

    @property
    def public_url(self) -> str:
        return self._public_url

    @property
    def port(self) -> int:
        return self._port


def _find_free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
