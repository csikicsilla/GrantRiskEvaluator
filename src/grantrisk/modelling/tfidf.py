"""TF-IDF settings and the vectoriser (SPEC-M1-05).

The vectoriser is never fitted here: M2 fits a fresh one on the training part of
each fold (SPEC-M2-03).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

# The Hungarian stop words of the old code line, cleaned (ISS-43): the two-word entry
# "az a" is dropped, because a stop word must be a single token, and so are the
# duplicates of "is" and "tehát".
HU_STOP_WORDS = (
    # articles
    "a", "az", "egy",
    # conjunctions
    "és", "vagy", "hogy", "de", "mint", "ha", "s", "illetve", "valamint",
    "továbbá", "azonban", "viszont", "tehát", "ezért", "mert", "míg", "sem",
    "se", "is", "csak", "csupán",
    # negation
    "nem", "ne", "sose", "soha",
    # verbal prefixes
    "meg", "el", "be", "ki", "fel", "le", "át", "rá", "ide", "oda",
    "össze", "szét", "vissza", "elő", "utána",
    # pronouns
    "ez", "azok", "ezek", "ő", "ők", "aki", "ami", "amely",
    "amelyek", "aminek", "amelynek", "ennek", "annak", "ezt", "azt",
    "ezen", "azon", "minden", "mindegyik", "valamennyi", "néhány",
    "bármely", "bármi", "senki", "semmi",
    # postpositions
    "által", "miatt", "szerint", "alapján", "részére", "számára",
    "esetén", "érdekében", "keretében", "vonatkozásában", "tekintetében",
    "között", "alatt", "felett", "mellett", "helyett", "nélkül", "után",
    "előtt", "során", "belül", "kívül", "iránt", "felé",
    # forms of "to be" and auxiliaries
    "van", "vannak", "volt", "voltak", "lesz", "lesznek", "lehet",
    "kell", "kellett", "lett", "legyen",
    # other frequent function words
    "úgy", "így", "itt", "ott", "akkor", "amikor", "ahol", "ahogy",
    "mind", "már", "még", "pedig", "hanem", "sőt", "azaz", "vagyis",
    "amennyiben", "illetőleg",
)

# \w matches every Unicode letter, so á, é, ő, ű … stay inside a token.
TOKEN_PATTERN = r"(?u)\b\w\w+\b"

DEFAULTS: dict[str, Any] = {
    "lowercase": True,
    "token_pattern": TOKEN_PATTERN,
    "ngram_range": [1, 2],
    "min_df": 2,
    "max_df": 0.9,
    "sublinear_tf": True,
    "max_features": 200_000,
    "stop_words": "hungarian",
}


def settings(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """The TF-IDF settings: the defaults of SPEC-M1-05, overridden by ``represent.tfidf``."""
    return {**DEFAULTS, **(values or {})}


def vectorizer(values: Mapping[str, Any] | None = None):
    """A new, unfitted vectoriser with the given settings; no stemming or lemmatisation."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    s = settings(values)
    stop = s["stop_words"]
    return TfidfVectorizer(
        lowercase=s["lowercase"],
        token_pattern=s["token_pattern"],
        ngram_range=tuple(s["ngram_range"]),
        min_df=s["min_df"],
        max_df=s["max_df"],
        sublinear_tf=s["sublinear_tf"],
        max_features=s["max_features"],
        stop_words=list(HU_STOP_WORDS) if stop == "hungarian" else stop,
        dtype=np.float64,
    )
