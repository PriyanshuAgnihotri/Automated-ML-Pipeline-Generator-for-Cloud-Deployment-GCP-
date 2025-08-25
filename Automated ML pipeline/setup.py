from setuptools import setup, find_packages

setup(
    name="automated-ml-pipeline",
    version="1.0.0",
    description="Automated ML Pipeline Generator for Cloud Deployment on GCP",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "google-cloud-storage",
        "google-cloud-aiplatform",
        "scikit-learn",
        "pandas",
        "numpy",
        "xgboost",
        "tensorflow",
        "pyyaml",
        "pytest"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)