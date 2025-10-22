"""
Setup script for blockchain-anomaly-detection package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_long_description():
    """Read the README.md file for the long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''


# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []


setup(
    name='blockchain-anomaly-detection',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A comprehensive system for detecting anomalies in blockchain transactions',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/arec1b0/blockchain-anomaly-detection',
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'examples']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.3.2',
            'pytest-cov>=4.1.0',
            'black>=23.3.0',
            'flake8>=6.0.0',
            'mypy>=1.3.0',
        ],
        'sentry': [
            'sentry-sdk>=1.25.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'blockchain-anomaly-detection=src.main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
