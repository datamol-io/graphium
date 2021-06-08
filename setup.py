from setuptools import setup
from setuptools import find_packages

setup(
    name="goli",
    version="0.1.0",
    author="Valence Discovery",
    author_email="dominique@valencediscovery.com",
    url="https://github.com/valence-discovery/goli",
    description="A deep learning library focused on graph representation learning for real-world chemical tasks.",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": ["goli=goli.cli:main_cli"],
    },
)
