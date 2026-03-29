import re

INJECTION_PATTERN = re.compile(
    r"ignore\s+(previous|all)\s+instructions?"
    r"|you\s+are\s+now"
    r"|forget\s+your\s+(previous|system)\s+(prompt|instructions?)"
    r"|act\s+as|pretend\s+(to\s+be|you\s+are)"
    r"|disregard|do\s+anything\s+now|\bDAN\b"
    r"|developer\s+mode|system\s*:|\[INST\]|</s>",
    re.IGNORECASE,
)

_TOPIC_KEYWORDS = [
    "aine", "kursus", "õppi", "loeng", "seminar", "õppekava",
    "ülikool", "semester", "eksamid", "baka", "magist", "tü",
    "course", "subject", "learn", "lecture", "university", "study",
    "exam",
]


def is_prompt_injection(text: str) -> bool:
    return bool(INJECTION_PATTERN.search(text))


def is_off_topic(text: str) -> bool:
    if len(text.split()) <= 10:
        return False
    lower = text.lower()
    return not any(kw in lower for kw in _TOPIC_KEYWORDS)
