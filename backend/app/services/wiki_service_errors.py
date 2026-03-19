"""Custom error types for WikiService."""


class WikiAlreadyExistsError(Exception):
    """Raised when attempting to generate a wiki that already exists."""

    def __init__(self, wiki_id: str) -> None:
        self.wiki_id = wiki_id
        super().__init__(f"Wiki already exists: {wiki_id}. Use refresh to regenerate.")
