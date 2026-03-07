"""
Arc Gateway — WebSocket control plane + WebChat UI.

Provides a persistent server that accepts messages via WebSocket,
routes them through the same Kernel/AgentLoop as the CLI, and
streams responses back. Includes a built-in WebChat interface.
"""
