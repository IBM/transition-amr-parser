import os
from setuptools import setup, find_packages

VERSION = '0.1.0'

package_data = {
    'transition_amr_parser': [
        'config.json',
        'entity_rules.json',
        'train.rules.json'
     ]
}

# this is what usually goes on requirements.txt
install_requires = [
    'torch==1.1.0',
    'h5py==2.10.0',
    'spacy==2.2.3',
    'tqdm==4.39.0',
    'fairseq==0.8.0'
]

setup(
    name='transition_amr_parser',
    version=VERSION,
    description="Trasition-based AMR parser tools",
    py_modules=['transition_amr_parser'],
    entry_points={
        'console_scripts': [
            'amr-learn = transition_amr_parser.learn:main',
            'amr-parse = transition_amr_parser.parse:main',
            'amr-oracle = transition_amr_parser.data_oracle:main',
            'amr-fake-parse = transition_amr_parser.fake_parse:main'
        ]
    },
    packages=find_packages(),
    install_requires=install_requires,
    package_data=package_data,
)
