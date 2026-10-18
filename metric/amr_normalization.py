# Copyright (c) 2026 Mirza Husadzic. SPDX-License-Identifier: MIT
# Added 2026-10-19; see ../LICENSES/fork-MIT.txt.
"""Shared role normalization for the local and global AMR scorers."""
from penman.models.amr import model

PROFILE = "amr-normalized-roles-v1"


def canonical_triple(triple):
    """Use PENMAN's aliases/inverses, then choose :mod for inverse :domain.

    This changes relation spelling/direction, not variable identity or graph top.
    Both :mod and :domain are canonical roles in PENMAN's AMR model; choosing
    one representation is our explicit profile extension.
    """
    source, role, target = model.deinvert(model.canonicalize(triple))
    return (target, ":mod", source) if role == ":domain" else (source, role, target)
