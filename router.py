"""Ollama Model Router - Sequential loading for RTX 4050 VRAM constraints."""

import asyncio
import time
import logging
from typing import Dict, Optional, Any, List
import ollama
from models import Signal, SignalAction, RegimeState
from config import load_config

logger = logging.getLogger(__name__)


class OllamaRouter:
    """
    Sequential model loading for 6GB VRAM constraint.
    Only ONE 8B-class model loaded at a time.
    phi3:3.8b kept warm for scalping.
    """
    
    # Model categories
    WARM_MODEL = "phi3:3.8b"           # Always loaded (<100ms)
    REASONING_MODELS = ["qwen3:8b"]    # Main reasoning
    DEEP_MODELS = ["deepseek-r1:8b"]   # Async macro
    EXECUTION_MODELS = ["mistral:7b"]  # Tool calling
    VISION_MODELS = ["moondream"]      # On-demand vision
    EMBEDDING_MODEL = "nomic-embed-text"
    
    # VRAM estimates (4-bit)
    VRAM_ESTIMATES = {
        "phi3:3.8b": 2.5,
        "qwen3:8b": 5.2,
        "deepseek-r1:8b": 5.2,
        "mistral:7b": 4.5,
        "moondream": 1.8,
        "nomic-embed-text": 0.5,
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.model_cfg = self.config.get("model", {})
        
        # Override from config if provided
        self.warm_model = self.model_cfg.get("warm_model", self.WARM_MODEL)
        self.reasoning_model = self.model_cfg.get("reasoning_model", self.REASONING_MODELS[0])
        self.deep_model = self.model_cfg.get("deep_model", self.DEEP_MODELS[0])
        self.execution_model = self.model_cfg.get("execution_model", self.EXECUTION_MODELS[0])
        self.vision_model = self.model_cfg.get("vision_model", self.VISION_MODELS[0])
        self.embedding_model = self.model_cfg.get("embedding_model", self.EMBEDDING_MODEL)
        
        self.ollama_host = self.model_cfg.get("ollama_host", "http://localhost:11434")
        self.num_predict = self.model_cfg.get("num_predict", 512)
        self.temperature = self.model_cfg.get("temperature", 0.3)
        self.top_p = self.model_cfg.get("top_p", 0.9)
        
        # State
        self.currently_loaded: Optional[str] = None
        self.warm_model_loaded = False
        self.last_used: Dict[str, float] = {}
        self._client: Optional[ollama.AsyncClient] = None
        
        # Model aliases for routing
        self.model_aliases = {
            "warm": self.warm_model,
            "phi3": self.warm_model,
            "scalp": self.warm_model,
            "reasoning": self.reasoning_model,
            "qwen": self.reasoning_model,
            "deep": self.deep_model,
            "macro": self.deep_model,
            "execution": self.execution_model,
            "mistral": self.execution_model,
            "tools": self.execution_model,
            "vision": self.vision_model,
            "moondream": self.vision_model,
            "embedding": self.embedding_model,
        }
    
    @property
    def client(self) -> ollama.AsyncClient:
        if self._client is None:
            self._client = ollama.AsyncClient(host=self.ollama_host)
        return self._client
    
    async def initialize(self):
        """Pre-load warm model and embedding model."""
        logger.info("Initializing Ollama router...")
        
        # Load embedding model first (small)
        await self._ensure_loaded(self.embedding_model)
        
        # Load warm model (phi3)
        await self._ensure_loaded(self.warm_model)
        self.warm_model_loaded = True
        
        logger.info(f"Router initialized. Warm model: {self.warm_model}")
    
    def route(self, context: Dict[str, Any]) -> str:
        """
        Select model based on context.
        
        Priority:
        1. Explicit task override
        2. Mode-based routing
        3. Default to reasoning model
        """
        # Explicit task
        if "task" in context:
            task = context["task"]
            if task in self.model_aliases:
                return self.model_aliases[task]
        
        # Mode-based
        mode = context.get("mode", "normal")
        urgency = context.get("urgency", "normal")
        
        if mode == "scalp" or urgency == "high":
            return self.warm_model
        if mode == "deep":
            return self.deep_model
        if mode == "vision":
            return self.vision_model
        if mode == "execution":
            return self.execution_model
        
        return self.reasoning_model
    
    async def _ensure_loaded(self, model: str) -> bool:
        """Ensure model is loaded, handling VRAM constraints."""
        if self.currently_loaded == model:
            self.last_used[model] = time.time()
            return True
        
        # Check if we need to unload current
        if self.currently_loaded and self.currently_loaded != model:
            await self._unload_current()
        
        # Load new model
        try:
            logger.info(f"Loading model: {model}")
            start = time.time()
            
            # Warm-up request
            await self.client.generate(
                model=model,
                prompt="Ready",
                options={"num_predict": 1, "temperature": 0}
            )
            
            self.currently_loaded = model
            self.last_used[model] = time.time()
            logger.info(f"Model {model} loaded in {time.time() - start:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model}: {e}")
            return False
    
    async def _unload_current(self):
        """Unload current model (Ollama doesn't have explicit unload, so we track)."""
        if self.currently_loaded:
            logger.debug(f"Unloading {self.currently_loaded}")
            self.currently_loaded = None
    
    async def generate(
        self,
        model: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        images: Optional[List[str]] = None,
        options: Optional[Dict] = None
    ) -> str:
        """
        Generate response from model.
        Handles model loading automatically.
        """
        # Ensure model is loaded
        await self._ensure_loaded(model)
        
        # Build options
        gen_options = {
            "num_predict": self.num_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if options:
            gen_options.update(options)
        
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            start = time.time()
            
            if images:
                response = await self.client.generate(
                    model=model,
                    prompt=prompt,
                    images=images,
                    system=system,
                    options=gen_options
                )
            else:
                response = await self.client.chat(
                    model=model,
                    messages=messages,
                    options=gen_options
                )
            
            elapsed = time.time() - start
            logger.debug(f"{model} inference: {elapsed:.2f}s")
            
            # Extract content
            if isinstance(response, dict):
                return response.get("message", {}).get("content", "") or response.get("response", "")
            return str(response)
            
        except Exception as e:
            logger.error(f"Generation failed for {model}: {e}")
            raise
    
    async def generate_signal(
        self,
        context: Dict[str, Any],
        system_prompt: str,
        user_prompt: str
    ) -> Signal:
        """
        Generate trading signal from context.
        Returns parsed Signal object.
        """
        model = self.route(context)
        
        response = await self.generate(
            model=model,
            prompt=user_prompt,
            system=system_prompt,
            options={"num_predict": 512, "temperature": 0.3}
        )
        
        return self._parse_signal(response, model, context)
    
    def _parse_signal(self, response: str, model: str, context: Dict) -> Signal:
        """Parse model response into Signal object."""
        import json
        import re
        
        # Try JSON first
        try:
            data = json.loads(response)
            return Signal(
                action=SignalAction(data.get("action", "HOLD").upper()),
                confidence=float(data.get("confidence", 0.5)),
                size_pct=float(data.get("size_pct", 0.5)),
                entry_price=data.get("entry_price"),
                stop_loss=data.get("stop_loss"),
                take_profit=data.get("take_profit"),
                reasoning=data.get("reasoning", ""),
                regime=RegimeState(data.get("regime")) if data.get("regime") else None,
                model_used=model
            )
        except:
            pass
        
        # Try regex extraction
        action_match = re.search(r'(BUY|SELL|HOLD)', response.upper())
        confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', response, re.IGNORECASE)
        size_match = re.search(r'size[:\s]+([0-9.]+)', response, re.IGNORECASE)
        reasoning_match = re.search(r'reasoning[:\s]+(.+)', response, re.IGNORECASE)
        
        return Signal(
            action=SignalAction(action_match.group(1)) if action_match else SignalAction.HOLD,
            confidence=float(confidence_match.group(1)) if confidence_match else 0.5,
            size_pct=float(size_match.group(1)) if size_match else 0.5,
            reasoning=reasoning_match.group(1).strip() if reasoning_match else response[:200],
            model_used=model
        )
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using nomic-embed-text."""
        await self._ensure_loaded(self.embedding_model)
        
        embeddings = []
        for text in texts:
            response = await self.client.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            embeddings.append(response["embedding"])
        
        return embeddings
    
    async def analyze_chart(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Analyze chart image using vision model."""
        default_prompt = (
            "Analyze this trading chart. Identify: "
            "1. Trend direction (up/down/sideways) "
            "2. Key support and resistance levels "
            "3. Breakout zones "
            "4. Volume profile "
            "5. Chart patterns (triangles, flags, wedges, etc.) "
            "Return as structured JSON."
        )
        
        return await self.generate(
            model=self.vision_model,
            prompt=prompt or default_prompt,
            images=[image_path],
            options={"num_predict": 1024, "temperature": 0.2}
        )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get router status."""
        return {
            "current_model": self.currently_loaded,
            "warm_model": self.warm_model,
            "warm_loaded": self.warm_model_loaded,
            "last_used": self.last_used,
            "vram_estimate_gb": self.VRAM_ESTIMATES.get(self.currently_loaded, 0),
        }
    
    async def shutdown(self):
        """Cleanup."""
        self._client = None
        logger.info("Ollama router shutdown")


# Convenience function
async def create_router(config_path: str = "config.yaml") -> OllamaRouter:
    router = OllamaRouter(config_path)
    await router.initialize()
    return router