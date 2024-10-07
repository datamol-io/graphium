from setuptools import setup
from setuptools.command.install import install
import subprocess
import sys

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)  # Run the default install process
        
        # Post-install logic here
        print("Running post-install script...")
        subprocess.check_call([sys.executable, './scripts/post_install.py'])  # Replace with your script or command

# Minimal setup
setup(
    cmdclass={
        'install': PostInstallCommand,
    }
)