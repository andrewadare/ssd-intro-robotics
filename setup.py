import codecs
from setuptools import setup, find_packages
import json
import os
import re

# Read json package info file to single-source the
# metadata, url, author, etc.  The version is
# parsed out of __init__.py directly.
here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'project_info.json'), 'r') as f:
    _info = json.load(f)


def read(mod, filename):
    return codecs.open('/'.join((here, mod, filename)), 'r').read()


def find_version(mod, filename):
    version_file = read(mod, filename)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='ssd-intro-robotics',
    version=find_version(mod='ssd_robotics', filename='__init__.py'),
    packages=find_packages(),
    url=_info['url'],
    license=_info['license'],
    author=_info['author'],
    author_email=_info['email'],
    description=_info['description'],
    install_requires=['filterpy', 'gr', 'numpy', 'scipy'],
)
