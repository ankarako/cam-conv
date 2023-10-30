import setuptools

setuptools.setup(
    name="camconv",
    version="1.0.0",
    author="Antonis Karakottas",
    description="A tiny library for converting between most major 3d learning library coordinate systems.",
    classifiers=["Programming Language :: Python :: 3", "License :: MIT"],
    python_requires=">=3.10",
    install_requires=[
        "numpy"
    ]
)