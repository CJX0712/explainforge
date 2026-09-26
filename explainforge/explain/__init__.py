"""Explainers: model-agnostic feature attribution.

Backends:
  * kernel_shap    — pure numpy KernelSHAP (offline default, SOTA-equivalent)
  * lime           — pure numpy LIME (weighted ridge on perturbations)
  * permutation    — scikit-learn permutation importance (standard method)
  * gradient_input — finite-difference gradient*input (classic saliency baseline)
  * shap_ref       — optional ``shap`` KernelExplainer reference (validates numpy impl)
All implement ExplainerProtocol and expose available_*() for graceful degradation.
"""
from __future__ import annotations

from .kernel_shap import KernelShapExplainer
from .lime import LimeExplainer
from .permutation import PermutationExplainer
from .gradient_input import GradientInputExplainer
from .shap_backend import ShapRefExplainer, SHAP_AVAILABLE
from .registry import REGISTRY, build, available

__all__ = [
    "KernelShapExplainer", "LimeExplainer", "PermutationExplainer",
    "GradientInputExplainer", "ShapRefExplainer", "SHAP_AVAILABLE",
    "REGISTRY", "build", "available",
]
