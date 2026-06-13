"""Canonical vocabulary for platforms and content types.

Single source of truth for the set of social platforms and content types the
project understands. Values are read from ``config.yaml`` (``data.platforms`` /
``data.content_types``) so the vocabulary can be changed in one place; the
literals here are only a fallback used when the config file is absent. Import
``PLATFORMS`` / ``CONTENT_TYPES`` (and the ``*_label`` helpers for UI) instead of
re-literalling these lists anywhere in the codebase.
"""

from typing import Dict, List

from .utils.config import get_config

_config = get_config()

PLATFORMS: List[str] = list(
    _config.get("data.platforms")
    or ["linkedin", "instagram", "twitter", "substack", "threads"]
)

CONTENT_TYPES: List[str] = list(
    _config.get("data.content_types") or ["no_video", "video"]
)

# Human-facing labels for UI dropdowns. A name like "LinkedIn" cannot be derived
# from ``str.title()`` (which yields "Linkedin"), so display strings live here in
# one place; an unknown platform falls back to a title-cased value.
_PLATFORM_LABELS: Dict[str, str] = {
    "linkedin": "LinkedIn",
    "instagram": "Instagram",
    "twitter": "Twitter",
    "substack": "Substack",
    "threads": "Threads",
}


def platform_label(platform: str) -> str:
    """Return the display label for a platform value."""
    return _PLATFORM_LABELS.get(platform, platform.replace("_", " ").title())


def content_type_label(content_type: str) -> str:
    """Return the display label for a content-type value."""
    return content_type.replace("_", " ").title()
