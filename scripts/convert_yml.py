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

"""
# Replace `ENV_FILE` with the path to the env file in the `requirements` folder
# which matches your machine's OS
Convert the dependencies from conda's ENV_FILE to pip `requirements.txt`
"""

import ruamel.yaml

yaml = ruamel.yaml.YAML()
data = yaml.load(open("ENV_FILE"))

requirements = []
for dep in data["dependencies"]:
    if isinstance(dep, str):
        outputs = dep.split("=")
        if len(outputs) == 1:
            package = outputs[0]
            requirements.append(package)
        elif len(outputs) == 2:
            package, package_version = outputs[0], outputs[1]
            requirements.append(package + "==" + package_version)
        elif len(outputs) == 3:
            package, package_version, python_version = outputs[0], outputs[1], outputs[2]
            requirements.append(package + "==" + package_version)
    elif isinstance(dep, dict):
        for preq in dep.get("pip", []):
            requirements.append(preq)

with open("requirements.txt", "w") as fp:
    for requirement in requirements:
        print(requirement, file=fp)
