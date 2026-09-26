"""ExplainForge error taxonomy (E100-E199).

Every public failure mode maps to a stable, greppable code so CI and downstream
consumers can branch on cause instead of parsing messages.
"""
from __future__ import annotations


class ExplainForgeError(Exception):
    code = "E100"
    message = "generic error"

    def __init__(self, message: str = ""):
        self.message = message or self.message
        super().__init__(f"[{self.code}] {self.message}")


class ConfigError(ExplainForgeError):
    code = "E101"
    message = "configuration error"


class DataError(ExplainForgeError):
    code = "E102"
    message = "data error"


class ModelError(ExplainForgeError):
    code = "E103"
    message = "model error"


class ExplainError(ExplainForgeError):
    code = "E104"
    message = "explainer error"


class MetricError(ExplainForgeError):
    code = "E105"
    message = "metric error"


class BackendUnavailableError(ExplainForgeError):
    code = "E106"
    message = "optional backend unavailable"
