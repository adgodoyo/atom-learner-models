"""Binding: shared LoRA basis, hypernetwork, global-mixing control and K-sample uncertainty helpers."""

from atom.binding.basis import LoRABasis
from atom.binding.derangement import assert_derangement, derangement_permutation
from atom.binding.global_mixing import GlobalMixingVector
from atom.binding.hooked_backbone import (LoRAHookedLinear, MixingContext, attach_lora_basis,
                                          build_second_basis, set_active_basis)
from atom.binding.hypernetwork import Hypernetwork
from atom.binding.uncertainty import (build_structured_committee_eps, mean_prediction,
                                      person_epistemic_spread, reparameterize,
                                      reparameterize_from_log_sigma)

__all__ = [
    "LoRABasis",
    "GlobalMixingVector",
    "Hypernetwork",
    "LoRAHookedLinear",
    "MixingContext",
    "attach_lora_basis",
    "build_second_basis",
    "set_active_basis",
    "derangement_permutation",
    "assert_derangement",
    "build_structured_committee_eps",
    "mean_prediction",
    "person_epistemic_spread",
    "reparameterize",
    "reparameterize_from_log_sigma",
]
