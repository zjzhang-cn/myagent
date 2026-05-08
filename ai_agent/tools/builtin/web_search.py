"""
网络搜索与网页抓取工具
"""

import re

from ai_agent.tools.base import tool


@tool(
    name="search_web",
    description="搜索互联网获取信息（基于 DuckDuckGo 纯文本搜索）",
    params=[
        {"name": "query", "type": "string", "description": "搜索关键词", "required": True},
        {"name": "max_results", "type": "number", "description": "最多返回条数", "required": False},
    ],
)
def search_web(query: str, max_results: int = 5) -> str:
    """搜索互联网，返回结果摘要"""
    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://html.duckduckgo.com/html/"
        resp = requests.post(
            url,
            data={"q": query, "b": ""},
            headers={"User-Agent": "Mozilla/5.0 (compatible; AI-Agent/1.0)"},
            timeout=15,
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.select(".result")

        if not results:
            return f"搜索 '{query}' 未找到结果。"

        lines = [f"搜索 '{query}' 的结果:"]
        for i, result in enumerate(results[:max_results]):
            title_el = result.select_one(".result__title")
            snippet_el = result.select_one(".result__snippet")
            link_el = result.select_one(".result__url")

            title = title_el.get_text(strip=True) if title_el else "无标题"
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            link = link_el.get_text(strip=True) if link_el else ""

            title = re.sub(r'\s+', ' ', title).strip()
            snippet = re.sub(r'\s+', ' ', snippet).strip()

            lines.append(f"\n{i+1}. {title}")
            if snippet:
                lines.append(f"   {snippet[:300]}")
            if link:
                lines.append(f"   URL: {link}")

        return "\n".join(lines)

    except ImportError:
        return (
            "搜索功能需要安装依赖: pip install requests beautifulsoup4 --break-system-packages"
        )
    except Exception as e:
        return f"搜索失败: {e}"


@tool(
    name="fetch_url",
    description="获取指定 URL 的网页内容（纯文本）",
    params=[
        {"name": "url", "type": "string", "description": "要抓取的 URL", "required": True},
        {"name": "max_chars", "type": "number", "description": "最大返回字符数", "required": False},
    ],
)
def fetch_url(url: str, max_chars: int = 8000) -> str:
    """抓取网页纯文本内容"""
    try:
        import requests
        from bs4 import BeautifulSoup

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AI-Agent/1.0)"},
            timeout=15,
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # 移除 script 和 style 标签
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n...(截断，原文共 {len(text)} 字符)"

        return f"URL: {url}\n\n{text}"

    except ImportError:
        return (
            "抓取功能需要安装依赖: pip install requests beautifulsoup4 --break-system-packages"
        )
    except Exception as e:
        return f"抓取失败: {e}"
