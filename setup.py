from setuptools import setup, find_packages

setup(
    name='deepchat',
    version='0.1',
    author="George Dittmar",
    packages=find_packages(),
    python_requires='>=3',
    package_data={},
    install_requires=["transformers",
                      "torch"]
)
