import os
import subprocess
from setuptools import setup, find_packages

VERSION = '0.4.0'

# this is what usually goes on requirements.txt
install_requires = [
    'torch==1.1.0',
    'h5py',
    # NOTE: For PPCs we wont have 2.2.3
    'spacy==2.2.3',
    'tqdm',
    'fairseq==0.8.0',
    # for scoring
    'smatch==1.0.4',
    # for debugging
    'ipdb',
    'line_profiler',
    'pyinstrument'
]

# You need to pip install the requirements.txt first
setup(
    name='transition_amr_parser',
    version=VERSION,
    description="Trasition-based AMR parser tools",
    py_modules=['transition_amr_parser', 'fairseq_ext'],
    entry_points={
        'console_scripts': [
            'amr-learn = transition_amr_parser.learn:main',
            'amr-parse = transition_amr_parser.parse:main',
            'amr-oracle = transition_amr_parser.data_oracle:main',
            'amr-fake-parse = transition_amr_parser.fake_parse:main',
            'amr-edit = transition_amr_parser.edit:main'
        ]
    },
    packages=find_packages(),
    install_requires=install_requires,
)
