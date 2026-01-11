"""Custom exceptions for RLM Extractor."""


class RLMExtractorError(Exception):
    """Base exception for RLM extraction errors."""

    pass


class APIKeyError(RLMExtractorError):
    """Raised when API keys are not configured."""

    pass


class SchemaError(RLMExtractorError):
    """Raised when JSON schema is invalid."""

    pass


class DocumentError(RLMExtractorError):
    """Raised when document is invalid or cannot be processed."""

    pass
