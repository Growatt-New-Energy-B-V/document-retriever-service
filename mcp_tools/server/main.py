"""FastMCP SSE server entrypoint for document tools."""
import logging
import os

from mcp.server.fastmcp import FastMCP

from .tools import register_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP("document-tools")
register_tools(mcp)

# Add a health endpoint for docker-compose healthcheck
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route


async def health(request):
    return JSONResponse({"status": "ok"})


# The FastMCP app is a Starlette app; we can add routes
app = mcp.sse_app()
app.routes.append(Route("/health", health))

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    logger.info(f"Starting MCP tools server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
