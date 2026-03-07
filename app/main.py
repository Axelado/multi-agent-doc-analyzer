"""Application FastAPI principale."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.db.session import init_db
from app.logging_config import setup_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle de l'application."""
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()

    logger.info("app_starting", host=settings.app_host, port=settings.app_port)

    # Initialiser la base de données
    await init_db()
    logger.info("database_initialized")

    yield

    logger.info("app_shutting_down")


def create_app() -> FastAPI:
    """Créer l'application FastAPI."""
    settings = get_settings()

    app = FastAPI(
        title="Plateforme NLP Multi-Agent",
        description=(
            "Plateforme locale d'analyse de rapports économiques/financiers "
            "avec architecture multi-agent anti-hallucination."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
