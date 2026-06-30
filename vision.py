"""Vision Pipeline - Chart analysis using Moondream (lightweight VLM)."""

import asyncio
import logging
from typing import Optional, Dict, Any
import base64
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VisionPipeline:
    """
    Lightweight chart analysis using Moondream.
    Only runs when chart screenshot is provided (NOT every candle).
    """
    
    def __init__(self, router):
        self.router = router
        self.enabled = True
    
    async def analyze_chart(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze trading chart screenshot.
        
        Returns structured analysis with:
        - trend direction
        - support/resistance levels
        - breakout zones
        - chart patterns
        - volume profile
        """
        if not self.enabled:
            return {"error": "Vision disabled"}
        
        default_prompt = (
            "Analyze this trading chart. Identify: "
            "1. Trend direction (up/down/sideways) "
            "2. Key support and resistance levels with prices "
            "3. Breakout zones "
            "4. Volume profile "
            "5. Chart patterns (triangles, flags, wedges, double top/bottom, head and shoulders) "
            "6. Moving average positions "
            "7. RSI/MACD visible signals "
            "Return as structured JSON with numeric price levels where possible."
        )
        
        try:
            # Use router's vision model
            response = await self.router.analyze_chart(
                image_path=image_path,
                prompt=prompt or default_prompt
            )
            
            return self._parse_vision_response(response)
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return {"error": str(e)}
    
    def _parse_vision_response(self, response: str) -> Dict[str, Any]:
        """Parse vision model response into structured data."""
        import json
        import re
        
        # Try JSON first
        try:
            return json.loads(response)
        except:
            pass
        
        # Extract key information with regex
        result = {
            "trend": "unknown",
            "support_levels": [],
            "resistance_levels": [],
            "patterns": [],
            "breakout_zones": [],
            "volume_profile": "normal",
            "ma_position": "unknown",
            "rsi_signal": "neutral",
            "macd_signal": "neutral",
            "confidence": 0.5,
            "raw_response": response[:500]
        }
        
        # Extract trend
        trend_match = re.search(r'(uptrend|downtrend|sideways|bullish|bearish|trending up|trending down)', response, re.IGNORECASE)
        if trend_match:
            result["trend"] = trend_match.group(1).lower()
        
        # Extract price levels
        price_levels
        price_matches = re.findall(r'(\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', response)
        prices = [float(p.replace('$', '').replace(',', '')) for p in price_matches]
        
        if prices:
            result["support_levels"] = [p for p in prices if p < max(prices)]
            result["resistance_levels"] = [p for p in prices if p == max(prices) or p > min(prices)]
        
        # Extract patterns
        pattern_keywords = ['triangle', 'flag', 'wedge', 'head and shoulders', 'double top', 'double bottom', 'cup and handle']
        for kw in pattern_keywords:
            if kw in response.lower():
                result["patterns"].append(kw)
        
        # Extract volume
        if 'high volume' in response.lower() or 'volume spike' in response.lower():
            result["volume_profile"] = "high"
        elif 'low volume' in response.lower():
            result["volume_profile"] = "low"
        
        # Extract MA position
        if 'above ma' in response.lower() or 'above moving average' in response.lower():
            result["ma_position"] = "above"
        elif 'below ma' in response.lower() or 'below moving average' in response.lower():
            result["ma_position"] = "below"
        
        # Extract RSI
        if 'oversold' in response.lower() or 'rsi.*30' in response.lower():
            result["rsi_signal"] = "oversold"
        elif 'overbought' in response.lower() or 'rsi.*70' in response.lower():
            result["rsi_signal"] = "overbought"
        
        # Extract MACD
        if 'bullish crossover' in response.lower() or 'macd.*cross.*up' in response.lower():
            result["macd_signal"] = "bullish"
        elif 'bearish crossover' in response.lower() or 'macd.*cross.*down' in response.lower():
            result["macd_signal"] = "bearish"
        
        return result


class ChartCapture:
    """Capture chart screenshots from various sources."""
    
    @staticmethod
    def from_webdriver(driver, element_selector: str = "canvas") -> Optional[str]:
        """Capture chart from Selenium WebDriver."""
        try:
            element = driver.find_element("css selector", element_selector)
            png = element.screenshot_as_png
            
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                f.write(png)
                return f.name
        except Exception as e:
            logger.error(f"Chart capture failed: {e}")
            return None
    
    @staticmethod
    def from_file(image_path: str) -> Optional[str]:
        """Validate and return image path."""
        path = Path(image_path)
        if path.exists() and path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            return str(path)
        return None
    
    @staticmethod
    def from_base64(base64_data: str) -> Optional[str]:
        """Save base64 image to temp file."""
        try:
            import tempfile
            import base64
            
            # Remove data URL prefix if present
            if 'base64,' in base64_data:
                base64_data = base64_data.split('base64,')[1]
            
            img_data = base64.b64decode(base64_data)
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                f.write(img_data)
                return f.name
        except Exception as e:
            logger.error(f"Base64 image save failed: {e}")
            return None
    
    @staticmethod
    def generate_synthetic_chart(candles: list, indicators: dict = None) -> str:
        """Generate chart image from candle data (for testing)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        import tempfile
        
        try:
            # Extract data
            timestamps = [datetime.fromtimestamp(c['timestamp']/1000) for c in candles]
            opens = [c['open'] for c in candles]
            highs = [c['high'] for c in candles]
            lows = [c['low'] for c in candles]
            closes = [c['close'] for c in candles]
            volumes = [c['volume'] for c in candles]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Candlesticks
            for i in range(len(candles)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([timestamps[i], timestamps[i]], [lows[i], highs[i]], color=color, linewidth=1)
                ax1.plot([timestamps[i], timestamps[i]], [opens[i], closes[i]], color=color, linewidth=4)
            
            # Moving averages if available
            if indicators:
                if 'ema_9' in indicators:
                    ax1.plot(timestamps, indicators['ema_9'], label='EMA 9', alpha=0.7)
                if 'ema_21' in indicators:
                    ax1.plot(timestamps, indicators['ema_21'], label='EMA 21', alpha=0.7)
            
            ax1.set_title('Price Chart')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volume
            colors = ['green' if closes[i] >= opens[i] else 'red' for i in range(len(candles))]
            ax2.bar(timestamps, volumes, color=colors, alpha=0.5, width=0.0005)
            ax2.set_title('Volume')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                plt.close()
                return f.name
                
        except Exception as e:
            logger.error(f"Synthetic chart generation failed: {e}")
            return None


# Convenience function
async def analyze_chart_image(router, image_path: str, custom_prompt: str = None) -> Dict[str, Any]:
    """Quick chart analysis helper."""
    pipeline = VisionPipeline(router)
    return await pipeline.analyze_chart(image_path, custom_prompt)