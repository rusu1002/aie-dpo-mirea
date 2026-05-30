import requests
import json
from typing import Optional, List, Dict
from src.utils.config import settings
from src.utils.logger import logger

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

AVAILABLE_MODELS = [
    "openai/gpt-120b:free",
    "google/gemmaa4b-it:free",
    "z-ai/glmair:free",
]

class ModelManager:
    """Управление моделями с proactive health check при старте"""
    
    def __init__(self):
        self.models = AVAILABLE_MODELS
        self.active_model: Optional[str] = None
        self.model_status: Dict[str, bool] = {}
        self.is_any_model_available = False
        
    def check_model_availability(self, model: str) -> bool:
        """Проверяет доступность модели без отправки полноценного запроса"""
        # Используем минимальный тестовый запрос
        test_message = [{
            "role": "user", 
            "content": "test"
        }]
        
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {settings.LLM_API_KEY}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": model,
                    "messages": test_message,
                    "max_tokens": 1,
                    "temperature": 0.0
                }),
                timeout=10  # Быстрый таймаут для проверки
            )
            
            if response.status_code == 200:
                logger.info(f"Model {model} is AVAILABLE")
                return True
            elif response.status_code == 429:
                logger.warning(f"Model {model} is RATE LIMITED")
                return False
            else:
                logger.warning(f"Model {model} returned status {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            logger.warning(f"Model {model} check TIMEOUT")
            return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"Model {model} check FAILED: {e}")
            return False
    
    def initialize(self) -> str:
        """Инициализирует менеджер и выбирает первую доступную модель"""
        logger.info("=" * 50)
        logger.info("CHECKING LLM MODELS AVAILABILITY")
        logger.info("=" * 50)

        available_models_found = False
        
        for model in self.models:
            logger.info(f"Testing model: {model}")
            if self.check_model_availability(model):
                self.active_model = model
                self.model_status[model] = True
                available_models_found = True
                logger.info(f"SELECTED MODEL: {model}")
                logger.info("=" * 50)
                break
            else:
                self.model_status[model] = False
        
        self.is_any_model_available = available_models_found

        # Если ни одна модель не доступна, используем первую как fallback (надо поменять на новый fallback с формированием ответа из предложений)
        if not available_models_found:
            logger.error("NO AVAILABLE MODELS FOUND!")
            logger.info("Will use fallback generation from context")
            self.active_model = self.models[0]
            logger.info("=" * 50)

        return self.active_model
    
    def get_active_model(self) -> str:
        """Возвращает текущую активную модель"""
        if not self.active_model:
            self.initialize()
        return self.active_model
    
    def is_llm_available(self) -> bool:
        """Возвращает, доступна ли хотя бы одна LLM модель"""
        if not self.active_model:
            self.initialize()
        return self.is_any_model_available
    
    def get_models_status(self) -> Dict:
        """Возвращает статус всех моделей"""
        return {
            "active_model": self.active_model,
            "models": self.model_status,
            "llm_available": self.is_any_model_available
        }

model_manager = ModelManager()