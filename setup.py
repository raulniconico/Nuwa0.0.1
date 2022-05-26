import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Nuwa",
    version="2.0.0",
    author="Zixing QIU",
    author_email="zixing.qiu@etu.enseeiht.fr",
    description="Ottergrad is an automatic differentiation tool support plenty of NumPy functions who borns from Nuwa "
                "framework. This project separates the auto-derivative function from Nuwa into a package, "
                "whose algorithm is more efficient, simpler and more stable than Nuwa 0.0.2.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raulniconico/Nuwa0.0.1",
    project_urls={
        "Bug Tracker": "https://github.com/raulniconico/Nuwa0.0.1",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"   ": "Nuwa"},
    packages=setuptools.find_packages(where="Nuwa"),
    python_requires=">=3.6",
)

