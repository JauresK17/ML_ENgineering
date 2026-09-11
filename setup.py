from setuptools import find_packages , setup 
from typing import List 

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    """Cettte fonction renvoie la liste des exigeance"""
    with open(file_path) as file_obj:
        lines = [line.strip() for line in file_obj]
    return [
        line for line in lines 
        if line and not line.startswith(("-","#"))
    ]



setup(
    name='mlproject',
    version='0.0.1',
    author='Kouame',
    author_email='kouamelaffind@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)