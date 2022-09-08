"""
Convert the dependencies from conda's `env.yml` to pip `requirements.txt`
"""

import ruamel.yaml

yaml = ruamel.yaml.YAML()
data = yaml.load(open("env.yml"))

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
