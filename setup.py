from setuptools import setup, find_packages

setup(
    name="my_project",  # Replace with your project's name
    version="1.0.0",  # Initial version
    description="A project integrating an agent manager with ChromaDB memory.",
    author="Your Name",  # Replace with your name
    author_email="your_email@example.com",  # Replace with your email
    packages=find_packages(where="src"),  # Locate packages under the src directory
    package_dir={"": "src"},  # Root package directory is src
    install_requires=[
        "chromadb",  # Add required dependencies here
        "langchain",
        "openai",
    ],
    python_requires=">=3.7",  # Ensure the Python version is compatible
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
