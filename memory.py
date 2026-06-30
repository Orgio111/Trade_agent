"""Local Memory + RAG using ChromaDB with Ollama embeddings."""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
import json
import ollama
from models import Trade
from pathlib import Path

logger = logging.getLogger(__name__)


class OllamaEmbeddingFunction:
    """Wrapper to use Ollama for embeddings with ChromaDB."""
    
    def __init__(self, model_name: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.client = ollama.Client(host=host)
    
    def name(self) -> str:
        """Return the name of the embedding function for ChromaDB."""
        return f"ollama-{self.model_name}"
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts. ChromaDB expects 'input' parameter."""
        embeddings = []
        for text in input:
            try:
                response = self.client.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                embeddings.append(response["embedding"])
            except Exception as e:
                logger.error(f"Embedding failed for text: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * 768)
        return embeddings


class LocalMemory:
    """
    Vector memory + pattern retrieval using ChromaDB with Ollama embeddings.
    Local, no cloud dependencies.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        import yaml
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        mem_cfg = self.config.get("memory", {})
        self.chroma_path = mem_cfg.get("chroma_path", "./memory")
        self.buffer_size = mem_cfg.get("buffer_size", 200)
        self.top_k = mem_cfg.get("similarity_top_k", 5)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Ollama embedding function
        model_cfg = {}
        try:
            with open("config.yaml") as f:
                import yaml
                full_config = yaml.safe_load(f)
                model_cfg = full_config.get("model", {})
        except:
            pass
        
        ollama_host = model_cfg.get("ollama_host", "http://localhost:11434")
        embedding_model = model_cfg.get("embedding_model", "nomic-embed-text")
        
        self.embedding_fn = OllamaEmbeddingFunction(
            model_name=embedding_model,
            host=ollama_host
        )
        
        # Collections
        self.trades_collection = self.client.get_or_create_collection(
            name="trades",
            embedding_function=self.embedding_fn,
            metadata={"description": "Trade history for pattern retrieval"}
        )
        
        self.patterns_collection = self.client.get_or_create_collection(
            name="patterns",
            embedding_function=self.embedding_fn,
            metadata={"description": "Market patterns and outcomes"}
        )
        
        self.regime_collection = self.client.get_or_create_collection(
            name="regimes",
            embedding_function=self.embedding_fn,
            metadata={"description": "Regime-specific patterns"}
        )
        
        logger.info(f"Memory initialized at {self.chroma_path} with Ollama embeddings")
    
    def store_trade(self, trade: Trade) -> str:
        """Store completed trade for future pattern matching."""
        doc = (
            f"Symbol: {trade.symbol} | Side: {trade.side} | "
            f"Entry: {trade.entry_price:.2f} | Exit: {trade.exit_price:.2f} | "
            f"Size: {trade.size:.4f} | PnL: {trade.pnl:.2f} ({trade.pnl_pct:.2%}) | "
            f"Regime: {trade.regime.value if trade.regime else 'unknown'} | "
            f"Model: {trade.model_used} | "
            f"Confidence: {trade.signal_confidence:.2f} | "
            f"Reason: {trade.reasoning}"
        )
        
        trade_id = f"trade_{trade.exit_time}"
        
        self.trades_collection.add(
            documents=[doc],
            metadatas=[trade.to_meta()],
            ids=[trade_id]
        )
        
        logger.debug(f"Stored trade: {trade_id}")
        return trade_id
    
    def store_pattern(self, 
                      regime: str,
                      setup: Dict[str, Any],
                      outcome: Dict[str, Any],
                      features: List[float]) -> str:
        """Store market pattern for regime-specific retrieval."""
        
        doc = (
            f"Regime: {regime} | "
            f"Setup: {json.dumps(setup)} | "
            f"Outcome: {json.dumps(outcome)}"
        )
        
        pattern_id = f"pattern_{regime}_{outcome.get('timestamp', 0)}"
        
        self.patterns_collection.add(
            documents=[doc],
            metadatas=[{
                "regime": regime,
                "setup": json.dumps(setup),
                "outcome": json.dumps(outcome),
                "features": json.dumps(features[:10])
            }],
            ids=[pattern_id]
        )
        
        return pattern_id
    
    def query_similar_trades(self,
                            current_regime: str,
                            symbol: str,
                            side: Optional[str] = None,
                            k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve similar historical trades for current setup.
        """
        k = k or self.top_k
        
        query = f"Symbol: {symbol} Regime: {current_regime}"
        if side:
            query += f" Side: {side}"
        query += " Trade setup similar to current"
        
        results = self.trades_collection.query(
            query_texts=[query],
            n_results=k,
            where={"symbol": symbol} if symbol else None
        )
        
        trades = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            trades.append({
                "document": doc,
                "metadata": meta,
                "similarity": 1.0
            })
        
        return trades
    
    def query_patterns(self,
                       regime: str,
                       setup_description: str,
                       k: int = None) -> List[Dict[str, Any]]:
        """
        Query patterns for specific regime.
        """
        k = k or self.top_k
        
        query = f"Regime: {regime} {setup_description}"
        
        results = self.patterns_collection.query(
            query_texts=[query],
            n_results=k,
            where={"regime": regime}
        )
        
        patterns = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            patterns.append({
                "document": doc,
                "metadata": meta,
            })
        
        return patterns
    
    def get_regime_stats(self, regime: str) -> Dict[str, Any]:
        """Get statistics for a specific regime."""
        results = self.trades_collection.query(
            query_texts=[f"Regime: {regime}"],
            n_results=100,
            where={"regime": regime}
        )
        
        if not results['metadatas'][0]:
            return {"trades": 0, "win_rate": 0, "avg_r": 0}
        
        wins = 0
        total_r = 0
        total = len(results['metadatas'][0])
        
        for meta in results['metadatas'][0]:
            pnl = meta.get('pnl', 0)
            if pnl > 0:
                wins += 1
            total_r += pnl / abs(meta.get('entry_price', 1)) if meta.get('entry_price', 1) != 0 else 0
        
        return {
            "trades": total,
            "win_rate": wins / total if total > 0 else 0,
            "avg_r": total_r / total if total > 0 else 0,
        }
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get most recent trades."""
        results = self.trades_collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        trades = []
        items = list(zip(results['documents'], results['metadatas']))
        items.sort(key=lambda x: x[1].get('exit_time', 0), reverse=True)
        
        for doc, meta in items[:limit]:
            trades.append({
                "document": doc,
                "metadata": meta
            })
        
        return trades
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "trades_count": self.trades_collection.count(),
            "patterns_count": self.patterns_collection.count(),
            "regimes_count": self.regime_collection.count(),
            "path": self.chroma_path,
        }