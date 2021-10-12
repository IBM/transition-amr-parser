from setuptools import setup, find_packages

VERSION = '0.5.1'

# this is what usually goes on requirements.txt
install_requires = [
    # 'torch',
    'torch==1.4',
    'h5py',
    'spacy==2.2.3',
    'tqdm',
    # 'fairseq',
    'fairseq==0.10.2',
    'tensorboardX',
    'packaging',
    # 'torch-scatter',
    # 'torch-scatter=1.3.2',
    # for scoring
    'penman',
    'smatch==1.0.4',
    # for debugging
    'ipdb',
    'line_profiler',
    'pyinstrument'
]

setup(
    name='transition_amr_parser',
    version=VERSION,
    description="Trasition-based AMR parser",
    py_modules=['transition_amr_parser'],
    entry_points={
        'console_scripts': [
            'amr-parse = transition_amr_parser.parse:main',
            'amr-machine = transition_amr_parser.amr_machine:main',
        ]
    },
    packages=find_packages(),
    install_requires=install_requires,
)
