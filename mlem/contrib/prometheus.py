from typing import Optional

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from mlem.contrib.fastapi import FastAPIMiddleware


class PrometheusFastAPIMiddleware(FastAPIMiddleware):
    instrumentator: Optional[Instrumentator]

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
