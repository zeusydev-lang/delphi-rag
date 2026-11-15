"""A question's names, ordinals and numbers: what a cached answer must agree on.

    signature("How did Gaius Gracchus die?")
    # Signature(names=("Caius Gracchus",), ordinals=(), numbers=())

The cache serves a stored answer only if the new question's signature equals
the stored one's. "Tiberius" vs "Caius", "first" vs "second" consulship, or
"133 BC" vs "121 BC" embed almost identically, so similarity alone can't tell
them apart; this can. It is deliberately strict: a missed cache hit costs
half a cent, a wrong cached answer costs trust.
"""

import re

from pydantic import BaseModel, ConfigDict

# Longest phrases first, so "Cato the Younger" wins over "Cato".
ALIASES: dict[str, str] = {
    "cato the elder": "Cato the Elder",
    "cato the censor": "Cato the Elder",
    "marcus cato": "Cato the Elder",
    "the censor": "Cato the Elder",
    "elder cato": "Cato the Elder",
    "cato the younger": "Cato the Younger",
    "cato of utica": "Cato the Younger",
    "younger cato": "Cato the Younger",
    "tiberius gracchus": "Tiberius Gracchus",
    "caius gracchus": "Caius Gracchus",
    "gaius gracchus": "Caius Gracchus",
    "elder gracchus": "Tiberius Gracchus",  # Tiberius was the elder by nine years
    "younger gracchus": "Caius Gracchus",
    "tiberius": "Tiberius Gracchus",
    "caius": "Caius Gracchus",
    "gaius": "Caius Gracchus",
    "gracchi": "Gracchi",  # both brothers
    "julius caesar": "Caesar",
    "pompey the great": "Pompey",
    "magnus": "Pompey",
}

# Capitalised words that aren't names.
NOT_NAMES = {
    "a", "an", "and", "are", "as", "at", "but", "by", "can", "could", "did", "do",
    "does", "for", "from", "had", "has", "have", "he", "her", "his", "how", "i",
    "if", "in", "is", "it", "of", "on", "or", "rome's", "said", "she", "so",
    "tell", "that", "the", "their", "then", "there", "they", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "whom", "whose", "why",
    "with", "would", "according", "describe", "explain", "give", "list", "name",
    "plutarch", "plutarch's",
}  # fmt: skip

ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}  # fmt: skip

_ALIAS_RX = re.compile(
    r"\b("
    + "|".join(re.escape(a) for a in sorted(ALIASES, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)
_ORDINAL_RX = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_NUMBER_RX = re.compile(r"\b(\d+)(?:\s*(BC|B\.C\.|AD|A\.D\.))?", re.IGNORECASE)
_WORD_RX = re.compile(r"[A-Za-z][A-Za-z'’]*")


class Signature(BaseModel):
    model_config = ConfigDict(frozen=True)

    names: tuple[str, ...] = ()
    ordinals: tuple[int, ...] = ()
    numbers: tuple[str, ...] = ()

    def differences(self, other: "Signature") -> list[str]:
        """Human-readable reasons two signatures don't match (empty if they do)."""
        out = []
        for field in ("names", "ordinals", "numbers"):
            mine, theirs = getattr(self, field), getattr(other, field)
            if mine != theirs:
                a = ", ".join(map(str, mine)) or "none"
                b = ", ".join(map(str, theirs)) or "none"
                out.append(f"{field}: {a} vs {b}")
        return out


def _possessive_off(word: str) -> str:
    return re.sub(r"['’]s?$", "", word)


def signature(question: str) -> Signature:
    text = question.replace("’", "'")
    names: set[str] = set()

    # 1. Known aliases, longest first; remove them so their words don't count twice.
    def take(m: re.Match) -> str:
        names.add(ALIASES[m.group(1).lower()])
        return " "

    text = _ALIAS_RX.sub(take, text)

    # 2. Ordinals ("first", "2nd") and numbers ("133 BC").
    ordinals = {int(m.group(1)) for m in _ORDINAL_RX.finditer(text)}
    text = _ORDINAL_RX.sub(" ", text)
    ordinals |= {
        ORDINAL_WORDS[w.lower()]
        for w in _WORD_RX.findall(text)
        if w.lower() in ORDINAL_WORDS
    }
    numbers = {
        f"{m.group(1)} {m.group(2).upper().replace('.', '')}"
        if m.group(2)
        else m.group(1)
        for m in _NUMBER_RX.finditer(text)
    }
    text = _NUMBER_RX.sub(" ", text)  # so "BC" isn't read as a name

    # 3. Any other capitalised word is a name (minus question words and the like).
    for word in _WORD_RX.findall(text):
        bare = _possessive_off(word)
        lowered = {bare.lower(), word.lower()}
        if bare[:1].isupper() and not lowered & (NOT_NAMES | ORDINAL_WORDS.keys()):
            names.add(bare)

    return Signature(
        names=tuple(sorted(names)),
        ordinals=tuple(sorted(ordinals)),
        numbers=tuple(sorted(numbers)),
    )
