from setuptools import setup, find_packages
from typing import List
# -e. to directly run setup.py to install the package
HYPHEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads a requirements file and returns a list of packages.

    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements=[req.replace("\n", "") for req in requirements]

# TO DO: Remove '-e .' if it exists
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="Manthan",
    author_email="manthanupadhyay11@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)