import codecs
import re

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))


def get_requirements(*parts):
    with codecs.open(path.join(here, *parts), 'r', 'utf8') as fp:
        return list(map(lambda x: x.strip(), fp.readlines()))


def read(*parts):
    with codecs.open(path.join(here, *parts), 'r', 'utf8') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

from setuptools import setup, find_packages

with codecs.open('README.md', 'r', 'utf8') as reader:
    long_description = reader.read()


with codecs.open('requirements.txt', 'r', 'utf8') as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))


setup(
    name='keras-lookahead',
    version=find_version('keras_lookahead', '__init__.py'),
    packages=find_packages(),
    url='https://github.com/CyberZHG/keras-lookahead',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@users.noreply.github.com',
    description='Lookahead mechanism for optimizers in Keras',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    install_requires=get_requirements('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
