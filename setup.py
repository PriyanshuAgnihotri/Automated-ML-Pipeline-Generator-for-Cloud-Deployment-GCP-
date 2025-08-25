
"""Config for installing a Python module/package."""

import setuptools
import ml_pipeline_gen

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='ml-pipeline-gen',
    version=ml_pipeline_gen.__version__,
    author='Michael Hu',
    author_email='author@example.com',
    description='A tool for generating end-to-end pipelines on GCP.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/GoogleCloudPlatform/ml-pipeline-generator-python',
    packages=['ml_pipeline_gen'],
    install_requires=[
        'cloudml-hypertune>=0.1.0.dev6',
        'gcsfs>=0.6.2',
        'google-api-python-client>=1.9.3',
        'google-cloud-container>=0.5.0',
        'jinja2>=2.11.2',
        'joblib>=0.15.1',
        'kfp>=0.5.1',
        'pandas>=1.0.4',
        'pyyaml>=5.3.1',
        'scikit-learn>=0.23.1',
        'tensorflow>=1.14.0,<2.0.0',
        'xgboost>=1.1.1',
    ],
    extras_require={
        'dev': [
            'mock',
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
