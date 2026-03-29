from setuptools import setup, find_packages

version = "0.1.0"

setup(
    name="anba4",
    version=version,
    description="Computes 6x6 stiffness and mass matrices of composite beam cross-sections",
    author="Marco Morandini, Basem Rajjoub",
    license="GNU GPL v3",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-fem",
    ],
    python_requires=">=3.8",
)
