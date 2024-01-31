from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='optimal_transport',
    version='1.0',
    description='Optimal transport tools for signal comparison and \
machine learning.',
    long_description='Optimal transport tools for signal comparison and \
machine learning.',
    author='George Strong',
    author_email='geowstrong@gmail.com',
    license='AGPL-3.0',
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=required)
