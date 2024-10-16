import platform
import sys
from pathlib import Path

import pkg_resources
from setuptools import find_packages, setup


requirements = []
if sys.platform.startswith("linux") and platform.machine() == "x86_64":
    requirements.append("triton>=2.0.0,<3")

setup(
    name="SWIM",
    py_modules=["swim"],
    version="2024.10.14",
    description="The official implementation of paper SWIM: SHORT-WINDOW CNN INTEGRATED WITH MAMBA FOR EEG-BASED AUDITORY SPATIAL ATTENTION DECODING",
    python_requires=">=3.9",
    author="Ziyang Zhang",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            Path(__file__).with_name("requirements.txt").open()
        )
    ],
)
