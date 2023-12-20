"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals, Graphcore Limited and Academic Collaborators.

This software is part of a collaboration between industrial and academic institutions.
Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals, Graphcore Limited, and its collaborators are not liable
for any damages arising from its use. Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


# Delete saved files like wandb, checkpoints, profiling etc.
# Usage: bash cleanup_files.sh
rm -rf wandb/*
rm -rf checkpoints/*
rm -rf pro_vision/*
rm -rf outputs/*
rm -rf models_checkpoints/*
rm -rf datacache/*
