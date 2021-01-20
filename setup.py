"""
goli
Implementing Tensor Networks
InVivo AI

"""
from setuptools import setup
import versioneer
import glob

short_description = __doc__.split("\n")

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:]),


setup(
    # Self-descriptive entries which should always be present
    name='goli',
    author='InVivo AI',
    author_email='dominique@invivoai.com',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='Not Open Source',
    scripts=glob.glob('bin/*'),

    # Which Python importable modules should be included when your package is installed
    packages=['goli'],

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='http://www.my_package.com',  # Website
    # install_requires=[],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    python_requires=">=3.5",
    install_requires=['numpy', 'pandas', 'torch', 'torchvision',
                        'tensorboardX']  # Python version restrictions
# 'rdkit', 'PIL'
    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)
