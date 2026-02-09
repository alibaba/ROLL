from setuptools import setup, find_packages

setup(
    name="roll",
    version="0.1.0",
    description="ROLL - Reinforcement Learning Optimization for Large-Scale Learning",
    packages=find_packages(include=["roll", "roll.*"]),
    python_requires=">=3.10",
)
