from setuptools import find_packages, setup

setup(
    name="us_visa",
    version="0.0.0",
    author="thameem",
    author_email="thameem4ever@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
