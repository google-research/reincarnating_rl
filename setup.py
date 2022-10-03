# coding=utf-8
# Copyright 2021 The Reincarnating RL Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup script for rliable.

This script will install reincarnating_rl code as a Python module.

See: https://github.com/google-research/reincarnating_rl
"""

import pathlib
from setuptools import find_packages
from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

install_requires = [
    'ale-py == 0.7.5',
    'AutoROM == 0.4.2',
    'AutoROM.accept-rom-license == 0.4.2',
    'dopamine-rl == 4.0.6',
]

description = (
    'Reincarnating RL: Open-source code for NeurIPS 2022 publication.')

setup(
    name='reincarnating_rl',
    version='1.0.0',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/google-research/reincarnating_rl',
    author='Rishabh Agarwal',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',

    ],
    keywords='benchmarking, research, reinforcement, machine, learning, research',
    include_package_data=True,
    packages=find_packages(exclude=['docs']),
    install_requires=install_requires,
    project_urls={  # Optional
        'Documentation': 'https://github.com/google-research/reincarnating_rl',
        'Bug Reports': 'https://github.com/google-research/reincarnating_rl/issues',
        'Source': 'https://github.com/google-research/reincarnating_rl',
    },
    license='Apache 2.0',
)
