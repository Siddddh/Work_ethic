"""
tinyfish_tool.py
----------------
LangChain Tool wrapper around the TinyFish web-agent API.

TinyFish handles browser automation, JS rendering, and anti-bot bypasses
so your agent can browse LinkedIn (and any other site) without being blocked.

Docs: https://docs.tinyfish.ai
"""

import json
import os
import time
from typing import Optional

import httpx
from langchain.tools import Tool


TINYFISH_API_URL = "https://api.tinyfish.ai/v1/agent"
TINYFISH_API_KEY = os.getenv("TINYFISH_API_KEY", "")


# ---------------------------------------------------------------------------
# Core TinyFish client
# ---------------------------------------------------------------------------

def run_tinyfish_task(
    task: str,
    start_url: Optional[str] = None,
    output_schema: Optional[dict] = None,
    timeout: int = 120,
) -> dict:
    """
    Send a task to TinyFish and return structured results.

    Args:
        task:          Natural-language instruction for the browser agent.
        start_url:     Optional URL to navigate to before executing the task.
        output_schema: Optional JSON schema describing the expected output shape.
        timeout:       Max seconds to wait for the task to complete.

    Returns:
        dict with keys: status, result, raw_text, url_visited
    """
    if not TINYFISH_API_KEY:
        raise ValueError(
            "TINYFISH_API_KEY environment variable is not set. "
            "Get your key at https://tinyfish.ai and set it with:\n"
            "  export TINYFISH_API_KEY=your_key_here"
        )

    headers = {
        "Authorization": f"Bearer {TINYFISH_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: dict = {"task": task}
    if start_url:
        payload["start_url"] = start_url
    if output_schema:
        payload["output_schema"] = output_schema

    with httpx.Client(timeout=timeout) as client:
        response = client.post(TINYFISH_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(
            f"TinyFish API error {response.status_code}: {response.text}"
        )

    data = response.json()
    return {
        "status": data.get("status", "unknown"),
        "result": data.get("result", {}),
        "raw_text": data.get("raw_text", ""),
        "url_visited": data.get("url_visited", start_url or ""),
    }


# ---------------------------------------------------------------------------
# Task templates
# ---------------------------------------------------------------------------

PROFILE_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "name":          {"type": "string",  "description": "Full name"},
        "title":         {"type": "string",  "description": "Current job title"},
        "company":       {"type": "string",  "description": "Current employer"},
        "location":      {"type": "string",  "description": "City / region"},
        "linkedin_url":  {"type": "string",  "description": "LinkedIn profile URL"},
        "email":         {"type": "string",  "description": "Email address if visible"},
        "about":         {"type": "string",  "description": "Summary / about section"},
        "connections":   {"type": "string",  "description": "Approx connection count"},
    },
    "required": ["name", "title", "company", "linkedin_url"],
}

ENRICHMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "email":         {"type": "string",  "description": "Best-guess email"},
        "phone":         {"type": "string",  "description": "Phone number if found"},
        "twitter":       {"type": "string",  "description": "Twitter/X handle"},
        "website":       {"type": "string",  "description": "Personal or company website"},
        "company_size":  {"type": "string",  "description": "Headcount range"},
        "industry":      {"type": "string",  "description": "Industry vertical"},
    },
}


def extract_linkedin_profile(linkedin_url: str) -> str:
    """Extract structured data from a single LinkedIn profile URL."""
    task = (
        f"Go to this LinkedIn profile: {linkedin_url}\n"
        "Extract the following fields and return them as JSON:\n"
        "- Full name\n"
        "- Current job title\n"
        "- Current company\n"
        "- Location\n"
        "- About / summary section\n"
        "- Email address (if visible in contact info)\n"
        "- LinkedIn profile URL\n"
        "If a field is not visible, use null."
    )
    result = run_tinyfish_task(
        task=task,
        start_url=linkedin_url,
        output_schema=PROFILE_EXTRACTION_SCHEMA,
    )
    return json.dumps(result["result"], indent=2)


def search_linkedin_leads(query: str) -> str:
    """
    Search LinkedIn for leads matching a natural-language query.
    Returns a JSON list of up to 10 profile summaries.
    """
    task = (
        f"Search LinkedIn for people matching: '{query}'.\n"
        "Use LinkedIn's People search at https://www.linkedin.com/search/results/people/\n"
        "Collect up to 10 results from the first page.\n"
        "For each person extract: full name, current title, current company, "
        "location, and their LinkedIn profile URL.\n"
        "Return a JSON array of objects with keys: name, title, company, location, linkedin_url."
    )
    result = run_tinyfish_task(
        task=task,
        start_url="https://www.linkedin.com/search/results/people/?keywords=" + query.replace(" ", "%20"),
        output_schema={
            "type": "array",
            "items": PROFILE_EXTRACTION_SCHEMA,
        },
    )
    # result["result"] may be a list or wrapped dict
    data = result["result"]
    if isinstance(data, dict):
        # TinyFish sometimes wraps arrays in {"items": [...]}
        data = data.get("items", data.get("results", [data]))
    return json.dumps(data, indent=2)


def enrich_lead(name: str, company: str, linkedin_url: str) -> str:
    """
    Enrich a lead with additional contact info using public sources.
    Tries Hunter.io pattern, company website contact page, etc.
    """
    task = (
        f"Find contact information for {name} who works at {company}.\n"
        f"Their LinkedIn is: {linkedin_url}\n"
        "Steps:\n"
        "1. Visit their LinkedIn profile and check the Contact Info section.\n"
        "2. Search Google for '{name} {company} email'.\n"
        "3. Try visiting the company's website and find a pattern for email addresses.\n"
        "Return any email address, phone number, Twitter handle, company website, "
        "company size, and industry you can find. Return as JSON."
    )
    result = run_tinyfish_task(
        task=task,
        output_schema=ENRICHMENT_SCHEMA,
    )
    return json.dumps(result["result"], indent=2)


# ---------------------------------------------------------------------------
# LangChain Tool definitions
# ---------------------------------------------------------------------------

def _search_tool_fn(query: str) -> str:
    """LangChain tool function: search LinkedIn for leads."""
    try:
        return search_linkedin_leads(query)
    except Exception as e:
        return f"Error searching LinkedIn: {e}"


def _extract_tool_fn(linkedin_url: str) -> str:
    """LangChain tool function: extract profile data from a URL."""
    linkedin_url = linkedin_url.strip().strip('"').strip("'")
    try:
        return extract_linkedin_profile(linkedin_url)
    except Exception as e:
        return f"Error extracting profile {linkedin_url}: {e}"


def _enrich_tool_fn(args: str) -> str:
    """
    LangChain tool function: enrich a lead with contact info.
    Input format: "name | company | linkedin_url"
    """
    try:
        parts = [p.strip() for p in args.split("|")]
        if len(parts) != 3:
            return "Error: provide input as 'name | company | linkedin_url'"
        name, company, linkedin_url = parts
        return enrich_lead(name, company, linkedin_url)
    except Exception as e:
        return f"Error enriching lead: {e}"


# ---------------------------------------------------------------------------
# Exported tools list — attach these to your LangChain agent
# ---------------------------------------------------------------------------

tinyfish_tools = [
    Tool(
        name="linkedin_search",
        func=_search_tool_fn,
        description=(
            "Search LinkedIn for people matching a query. "
            "Input: a natural-language description of the leads you want, "
            "e.g. 'CTO at SaaS startup in Bangalore' or 'VP Sales fintech London'. "
            "Returns a JSON list of matching profiles with name, title, company, location, linkedin_url."
        ),
    ),
    Tool(
        name="linkedin_extract_profile",
        func=_extract_tool_fn,
        description=(
            "Extract full profile data from a single LinkedIn URL. "
            "Input: a LinkedIn profile URL string, e.g. 'https://www.linkedin.com/in/username'. "
            "Returns JSON with name, title, company, location, about, email (if visible), linkedin_url."
        ),
    ),
    Tool(
        name="enrich_lead",
        func=_enrich_tool_fn,
        description=(
            "Enrich a lead with additional contact info (email, phone, social). "
            "Input MUST be in the format: 'Full Name | Company Name | LinkedIn URL'. "
            "Returns JSON with email, phone, twitter, website, company_size, industry."
        ),
    ),
]
