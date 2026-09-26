"""Explainer registry + availability probes (graceful degradation)."""
from __future__ import annotations

from ..core.errors import ExplainError
from .kernel_shap import KernelShapExplainer
from .lime import LimeExplainer
from .permutation import PermutationExplainer
from .gradient_input import GradientInputExplainer
from .shap_backend import ShapRefExplainer, SHAP_AVAILABLE

REGISTRY = {
    "kernel_shap": KernelShapExplainer,
    "lime": LimeExplainer,
    "permutation": PermutationExplainer,
    "gradient_input": GradientInputExplainer,
    "shap_ref": ShapRefExplainer,
}


def build(name: str, **kwargs):
    if name not in REGISTRY:
        raise ExplainError(f"unknown explainer: {name}")
    return REGISTRY[name](**kwargs)


def available(name: str) -> bool:
    if name == "shap_ref":
        return SHAP_AVAILABLE
    return name in REGISTRY
