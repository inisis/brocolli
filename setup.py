import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brocolli",
    version="3.0.0",
    author="desmond",
    author_email="desmond.yao@buaa.edu.cn",
    description="everything in pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/inisis/brocolli",
    install_requires=["loguru", "onnx==1.9.0", "onnxruntime==1.8.0", "tabulate"],
    project_urls={
        "Bug Tracker": "https://github.com/inisis/brocolli/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(exclude=["bin", "test", "imgs"]),
    python_requires=">=3.6",
    keywords="machine-learning, pytorch, caffe, torchfx",
)
