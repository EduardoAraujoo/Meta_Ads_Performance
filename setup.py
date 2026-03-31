from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Retorna a lista de requisitos do arquivo requirements.txt
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='ml_project_science',
    version='0.0.1',
    author='Marketing Science Architect',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
