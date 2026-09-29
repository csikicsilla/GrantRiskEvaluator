"""Document type from the file name, or from the first page (SPEC-C1-06)."""

from __future__ import annotations

import re
import unicodedata

MAIN_CALL = "main_call"
PRODUCT_DESCRIPTION = "product_description"
OTHER = "other"

# Rule 2: markers of annexes, templates, guides, grant deeds (-to = támogatói okirat) and forms.
NON_CALL = re.compile(
    r"utmutato|mellekl|sablon|minta|szerzod|okirat|aszf|segedlet|kerdoiv|adatlap|nyomtatvany"
    r"|eljarasi-rend|kezikonyv|tajekoztato|mutato|tsm|cash-flow|likviditasi|import"
    r"|(?:^|-)to(?:-|\.pdf$)"
)
TITLE_LINES = 8  # rule 5 looks for the title among the first non-empty lines of page 1


def fold(text: str) -> str:
    """Lower case without accents: 'Felhívás' → 'felhivas'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def by_name(original_name: str) -> str | None:
    """Rules 2–4. None means the name decides nothing and the first page must be read."""
    name = fold(original_name)
    if NON_CALL.search(name):
        return OTHER
    if "termekleir" in name:
        return PRODUCT_DESCRIPTION
    if "felhiv" in name:
        return MAIN_CALL
    return None


def by_first_page(page_text: str) -> str | None:
    """Rule 5: the type named in the title of page 1, or None if it names neither."""
    lines = [line for line in (fold(x).strip() for x in page_text.splitlines()) if line][:TITLE_LINES]
    # Without spaces, so that a drop capital ("F" / "ELHÍVÁS" on two lines) still reads as one word.
    title = re.sub(r"\s+", "", "".join(lines))
    if "termekleiras" in title:
        return PRODUCT_DESCRIPTION
    if "felhivas" in title:
        return MAIN_CALL
    return None
