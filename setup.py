from setuptools import setup, find_packages
from typing import List




# This function is used to get the list of requirements from the requirements.txt file
def get_requirements(file_path):
    """
    This function reads the requirements from a file and returns them as a list.
    """
    requirements= []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')



setup(
    name='Project_1',
    version='0.1',
    author="k Sriganesh",
    author_email="sriganeshkavali@gmail.com",
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')
)