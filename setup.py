from setuptools import setup
from setuptools import find_packages

# Sync the env.yml file here
install_requires = [
    "click",
    "tqdm",
    "loguru",
    "omegaconf",
    "appdirs",
    "fsspec>=2021.6",
    "s3fs>=2021.6",
    "gcsfs>=2021.6",
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "torch>=1.10",
    # "torchvision",
    "tensorboard",
    "pytorch-lightning>=1.9",
    "torchmetrics>=0.2",
    # "dgl>=0.5.2",
    "ogb",
    "datamol",
    "mordred",
    "umap-learn",
]

setup(
    name="goli",
    version="1.1.0",
    author="Valence Discovery",
    author_email="dominique@valencediscovery.com",
    url="https://github.com/valence-discovery/goli",
    description="A deep learning library focused on graph representation learning for real-world chemical tasks.",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/valence-discovery/goli/issues",
        "Documentation": "https://valence-discovery.github.io/goli",
        "Source Code": "https://github.com/valence-discovery/goli",
    },
    python_requires=">=3.7",  # datamol requires a minimal version of 3.8
    # install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": ["goli=goli.cli:main_cli"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
