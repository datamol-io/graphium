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


"""
Start the ipu environment and SDK
"""

source /opt/gc/sdk-3.0.0+1128/poplar-ubuntu_20_04-3.0.0+5468-0379b9a65d/enable.sh
source /opt/gc/sdk-3.0.0+1128/popart-ubuntu_20_04-3.0.0+5468-0379b9a65d/enable.sh

source ~/.venv/graphium_ipu/bin/activate # Change to your path

export VISUAL=vim
export EDITOR="$VISUAL"
