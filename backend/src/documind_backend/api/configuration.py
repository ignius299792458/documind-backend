"""
Configuration settings for the Documind settings
"""
import logging
from fastapi import HTTPException, status, APIRouter

import documind_backend.models.schemas as Schemas
from documind_backend.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


# ==============================================================================
# POST /admin/config/model-params
# ==============================================================================

@router.post(
   path="/admin/config/model-params",
   status_code=status.HTTP_226_IM_USED,
   response_model=Schemas.ConfigurationModelParameterResponse,
   summary="",
   description=""
)
async def configure_model_params(request: Schemas.ConfigurationModelParameterBody) -> Schemas.ConfigurationModelParameterResponse:
   settings.ollama_creativity_temperature = request.ollama_creativity_temperature
   settings.retrieval_top_k = request.retrieval_top_k
   settings.rerank_top_n = request.rerank_top_n
   settings.confidence_threshold = request.confidence_threshold
   
   return Schemas.ConfigurationModelParameterResponse(
      is_plugged=True
   )