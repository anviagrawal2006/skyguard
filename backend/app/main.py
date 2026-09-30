import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from app.core.database import engine, Base
from app.models import *  # This registers all models

from app.api.routes.stations import router as stations_router
from app.api.routes.readings import router as readings_router
from app.api.routes.health import router as health_router
from app.api.routes.anomalies import router as anomalies_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.corrections import router as corrections_router
from app.api.routes.predict import router as predict_router
from app.services.ingestion_service import poll_stations

# Create database tables (Alembic will handle this usually, but keep for fallback/dev)
# Wait, since TRD says "Generate an Alembic migration", we should let Alembic create them.
# But keeping it doesn't hurt if we use `Base.metadata.create_all(bind=engine)`.
# Let's rely on Alembic. I will comment it out or leave it for development.
# Base.metadata.create_all(bind=engine)

scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup scheduler on startup
    poll_interval = int(os.getenv("POLL_INTERVAL_MINUTES", "5"))
    scheduler.add_job(poll_stations, "interval", minutes=poll_interval)
    scheduler.start()
    
    # Initialize ML pipeline
    from app.services.ml_service import init_ml_pipeline
    init_ml_pipeline()
    
    yield
    # Shutdown scheduler on exit
    scheduler.shutdown()

app = FastAPI(
    title="SkyGuard AI API",
    lifespan=lifespan
)

# Support comma-separated origins for multi-environment CORS
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
allowed_origins = [origin.strip() for origin in frontend_url.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
api_router = APIRouter(prefix="/api")
api_router.include_router(stations_router)
api_router.include_router(readings_router)
api_router.include_router(health_router)
api_router.include_router(anomalies_router)
api_router.include_router(dashboard_router)
api_router.include_router(corrections_router)
api_router.include_router(predict_router)

app.include_router(api_router)

# Also adding extra routes required by TRD in their respective files...

@app.get("/")
def home():
    return {
        "message": "SkyGuard AI Backend is running"
    }