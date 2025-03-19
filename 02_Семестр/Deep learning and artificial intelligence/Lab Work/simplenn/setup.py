"""
Скрипт установки пакета SimpleNN.
"""

from setuptools import setup, find_packages

setup(
    name="simplenn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
    ],
    author="SimpleNN Team",
    author_email="example@example.com",
    description="Простой нейросетевой фреймворк для Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/simplenn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
)