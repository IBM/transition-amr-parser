import os
from setuptools import setup, find_packages

VERSION = '0.3.1'

package_data = {
    'transition_amr_parser': [
        'config.json',
        'entity_rules.json',
        'train.rules.json'
     ]
}

# this is what usually goes on requirements.txt
# Note that we use a concda installer for this 
# scripts/stack-transformer/ccc_x86_fairseq.yml
install_requires = [
    'torch==1.1.0',
    'h5py',
    'spacy==2.2.3',
    'tqdm'
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
            'amr-fake-parse = transition_amr_parser.fake_parse:main',
            'amr-edit = transition_amr_parser.edit:main'
        ]
    },
    packages=find_packages(),
    install_requires=install_requires,
    package_data=package_data,
)
