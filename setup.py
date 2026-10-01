from setuptools import setup, find_packages

setup(
    name="rga-pytorch",
    version="1.0.0",
    description="Recursive Gradient Absorption for deep neural networks",
    author="[YOUR NAME]",
    packages=find_packages(),
    install_requires=["torch>=1.9.0", "torchvision>=0.10.0"],
    python_requires=">=3.7",
)
