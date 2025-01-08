import os
from typing import List
from setuptools import setup, find_packages

"""
The Setup.py file is an essential part of packaging and distributing python
projects. Its is used by setuptools (or distutils in order python version)
to define the configuration of your project, such as its metadata, dependencies
, and more....
"""

def get_requirements()->List[str]:

    """
    This function wil return list of requirements 
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            #Read lines from the file
            lines=file.readlines()
            ## Process each line
            for line in lines:
                requirement=line.strip()
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

setup(
    name="FinancialRiskforLoanApproval",
    version="0.0.1",
    author="Vishnu",
    author_email="vishnurrajeev@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)