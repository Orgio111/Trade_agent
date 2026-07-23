from __future__ import annotations

import pytest

from scripts.apply_migrations import quote_role


def test_database_role_identifier_is_strictly_validated() -> None:
    assert quote_role("quantex_runtime") == '"quantex_runtime"'
    for invalid in ("", "runtime-admin", "runtime; DROP ROLE quantex", "9runtime"):
        with pytest.raises(ValueError):
            quote_role(invalid)
