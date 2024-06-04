from setuptools import setup, find_packages

setup(
    name='image_feature_extraction',
    version='0.1.0',
    description='A package for extracting various features from 2D images.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'scikit-image',
        'scikit-learn',
        'pywavelets',
        'gudhi'
    ],
)
