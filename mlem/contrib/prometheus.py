"""Instrumenting FastAPI app to expose metrics for prometheus
Extension type: middleware

Exposes /metrics endpoint
"""
from typing import ClassVar, Optional

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from mlem.contrib.fastapi import FastAPIMiddleware


class PrometheusFastAPIMiddleware(FastAPIMiddleware):
    """Middleware for FastAPI server that exposes /metrics endpoint to be scraped by Prometheus"""

    type: ClassVar = "prometheus_fastapi"

    instrumentator: Optional[Instrumentator]
    """Instrumentator instance to use. If not provided, a new one will be created"""

    class Config:
        arbitrary_types_allowed = True

    def on_app_init(self, app: FastAPI):
        @app.on_event("startup")
        async def startup():
            (self.instrumentator or Instrumentator()).instrument(app).expose(
                app
            )

    def on_init(self):
        pass

    def on_request(self, request):
        return request

    def on_response(self, request, response):
        return response
