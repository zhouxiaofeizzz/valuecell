"""News-related tools for the News Agent."""

import os
from datetime import datetime
from typing import Optional

from agno.agent import Agent
import httpx
from loguru import logger

from valuecell.adapters.models import create_model
from valuecell.config.manager import get_config_manager


async def web_search(query: str) -> str:
    """Search web for the given query and return a summary of the top results.

    This function uses the centralized configuration system to create model instances.
    It supports multiple search providers:
    - Google (Gemini with search enabled) - when WEB_SEARCH_PROVIDER=google and GOOGLE_API_KEY is set
    - Perplexity (via OpenRouter) - default fallback

    Args:
        query: The search query string.

    Returns:
        A summary of the top search results.
    """
    provider = os.getenv("WEB_SEARCH_PROVIDER", "").lower()
    if provider == "tavily" and os.getenv("TAVILY_API_KEY"):
        return await _web_search_tavily(query)
    if provider == "google" and os.getenv("GOOGLE_API_KEY"):
        return await _web_search_google(query)
    if os.getenv("TAVILY_API_KEY"):
        return await _web_search_tavily(query)
    if os.getenv("GOOGLE_API_KEY"):
        return await _web_search_google(query)

    # Use Perplexity Sonar via OpenRouter for web search
    # Perplexity models are optimized for web search and real-time information
    if not os.getenv("OPENROUTER_API_KEY"):
        return "Web search provider not configured"
    model = create_model(
        provider="openrouter",
        model_id="perplexity/sonar",
        max_tokens=None,
    )
    response = await Agent(model=model).arun(query)
    return response.content


async def _web_search_google(query: str) -> str:
    """Search Google for the given query and return a summary of the top results.

    Uses Google Gemini with search grounding enabled for real-time web information.

    Args:
        query: The search query string.

    Returns:
        A summary of the top search results.
    """
    # Use Google Gemini with search enabled
    # The search=True parameter enables Google Search grounding for real-time information
    model = create_model(
        provider="google",
        model_id="gemini-2.5-flash",
        search=True,  # Enable Google Search grounding
    )
    response = await Agent(model=model).arun(query)
    return response.content


async def _web_search_tavily(query: str) -> str:
    try:
        manager = get_config_manager()
        cfg = manager.get_provider_config("tavily")
        api_key = os.getenv("TAVILY_API_KEY") or (cfg.api_key if cfg else None)
        base_url = (
            cfg.base_url if cfg and cfg.base_url else "https://api.tavily.com"
        ).rstrip("/")
        if not api_key:
            return "Tavily API key not set"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 8,
            "include_raw_content": False,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(f"{base_url}/search", json=payload)
            r.raise_for_status()
            data = r.json()
        answer = data.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        results = data.get("results") or []
        if not results:
            return ""
        lines = []
        for item in results[:8]:
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            content = str(item.get("content", "")).strip()
            if title and url:
                lines.append(f"{title}\n{url}\n{content[:300]}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.error("Tavily search failed: {err}", err=str(e))
        return f"Error fetching Tavily results: {str(e)}"


async def get_breaking_news() -> str:
    """Get breaking news and urgent updates.

    Returns:
        Formatted string containing breaking news
    """
    try:
        search_query = "breaking news urgent updates today"
        logger.info("Fetching breaking news")

        news_content = await web_search(search_query)
        return news_content

    except Exception as e:
        logger.error(f"Error fetching breaking news: {e}")
        return f"Error fetching breaking news: {str(e)}"


async def get_financial_news(
    ticker: Optional[str] = None, sector: Optional[str] = None
) -> str:
    """Get financial and market news.

    Args:
        ticker: Stock ticker symbol for company-specific news
        sector: Industry sector for sector-specific news

    Returns:
        Formatted string containing financial news
    """
    try:
        search_query = "financial market news"

        if ticker:
            search_query = f"{ticker} stock news financial market"
        elif sector:
            search_query = f"{sector} sector financial news market"

        # Add time constraint for recent news
        today = datetime.now().strftime("%Y-%m-%d")
        search_query += f" {today}"

        logger.info(f"Searching for financial news with query: {search_query}")

        news_content = await web_search(search_query)
        return news_content

    except Exception as e:
        logger.error(f"Error fetching financial news: {e}")
        return f"Error fetching financial news: {str(e)}"
