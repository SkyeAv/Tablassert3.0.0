from setuptools import setup, find_packages

setup(
    name="Tablassert",
    version="3.0.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn", "requests", "openpyxl", "pyyaml", "pandas", "numpy",
        "xlrd", "nltk"],
    entry_points={
        "console_scripts": [
            "tablassert=src.main:main", "tablassert_test=src.test:main"]},
    description="""
        Tablassert is a multipurpose tool that crafts knowledge assertions
        from tabular data, augments knowledge with configuration, and exports
        knowledge as Knowledge Graph Exchange (KGX) consistent TSVs.""",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Skye Goetz (ISB)",
    author_email="sgoetz@systemsbiology.org",
    url="https://github.com/SkyeAv/Tablassert3.0.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    python_requires=">=3.12.6")
