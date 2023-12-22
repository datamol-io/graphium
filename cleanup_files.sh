"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals and Graphcore Limited.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals and Graphcore Limited are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
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
