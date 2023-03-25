"""Instrumenting FastAPI app to expose metrics for prometheus
Extension type: middleware

Exposes /metrics endpoint
"""
from typing import ClassVar, List, Optional

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from mlem.contrib.fastapi import FastAPIMiddleware
from mlem.utils.importing import import_string_with_local
from mlem.utils.module import get_object_requirements


class PrometheusFastAPIMiddleware(FastAPIMiddleware):
    """Middleware for FastAPI server that exposes /metrics endpoint to be scraped by Prometheus"""

    type: ClassVar = "prometheus_fastapi"

    metrics: List[str] = []
    """Instrumentator instance to use. If not provided, a new one will be created"""
    instrumentator_cache: Optional[Instrumentator] = None

    class Config:
        arbitrary_types_allowed = True
        exclude = {"instrumentator_cache"}

    @property
    def instrumentator(self):
        if self.instrumentator_cache is None:
            self.instrumentator_cache = self.get_instrumentator()
        return self.instrumentator_cache

    def on_app_init(self, app: FastAPI):
        @app.on_event("startup")
        async def _startup():
            self.instrumentator.expose(app)

    def on_init(self):
        pass

    def on_request(self, request):
        return request

    def on_response(self, request, response):
        return response

    def get_instrumentator(self):
        instrumentator = Instrumentator()
        for metric in self._iter_metric_objects():
            # todo: check object type
            instrumentator.add(metric)
        return instrumentator

    def _iter_metric_objects(self):
        for metric in self.metrics:
            # todo: meaningful error on import error
            yield import_string_with_local(metric)

    def get_requirements(self):
        reqs = super().get_requirements()
        for metric in self._iter_metric_objects():
            reqs += get_object_requirements(metric)
        return reqs
