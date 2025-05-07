from fastapi import APIRouter, HTTPException
from .schemas import NerRequest, NerResponse
from src.ner_service.models.predictor import NERPredictor
from .schemas import NerEntity
from loguru import logger

ner_predictor = NERPredictor()
router = APIRouter()


@router.post("/predict", response_model=NerResponse, summary="NER预测")
async def ner_prediction(request: NerRequest) -> NerResponse:
    """
    命名实体识别接口

    - **text**: 输入文本（必需）
    - **threshold**: 置信度阈值（可选，默认0.5）
    """
    logger.info(f"Received NER request for text: '{request.text[:50]}...'")
    try:
        entities = ner_predictor.predict(
            text=request.text,
            threshold=request.threshold if request.threshold is not None else 0.5
        )
        ner_entities = [NerEntity(**entity) for entity in entities]
        logger.info(f"Prediction successful, found {len(ner_entities)} entities.")
        return NerResponse(entities=ner_entities)
    except Exception as e:
        logger.error(f"Unhandled exception during NER prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during NER prediction.")
