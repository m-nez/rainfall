import setuptools

setuptools.setup(
    name="rain",
    version="0.0.1",
    author="Michał Nieznański",
    description="Get rain data from UW",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "easyocr",
        "matplotlib",
        "numpy",
        "fire",
        "requests",
        "pandas"
        ]
)
