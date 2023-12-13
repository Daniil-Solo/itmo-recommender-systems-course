from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.router import router
from src.exceptions import RecSysServiceError
from src.api.dependencies import get_actual_recommender
from src.config import settings, ModeEnum


@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Executes start actions
    """
    application.state.recommender = get_actual_recommender()
    yield


app = FastAPI(
    lifespan=lifespan if settings.mode == ModeEnum.PRODUCTION else None,
    title="Recommendation System Service",
    version="0.0.1",
    docs_url="/docs",
    redoc_url="/docs/redoc",
)

app.include_router(router)


@app.exception_handler(RecSysServiceError)
async def handle_rec_sys_service_errors(_: Request, exc: RecSysServiceError):
    """
    Handles service errors
    """
    return JSONResponse(
        content={"message": exc.error_message}, status_code=exc.status_code
    )
