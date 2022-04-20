from setuptools import setup, find_packages

VERSION = '0.5.1'

# this is what usually goes on requirements.txt
install_requires = [
    'torch==1.6',
    #'torch-scatter==1.3.2',
    'tqdm',
    'fairseq==0.10.2',
    'packaging',
    'requests',
    # for scoring
    'penman',
    'smatch==1.0.4',
    # for debugging
    'ipdb',
    'line_profiler',
    'pyinstrument',
    'sentencepiece',
    'tensorboardX',
]

setup(
    name='transition_amr_parser',
    version=VERSION,
    description="Trasition-based AMR parser",
    py_modules=['transition_amr_parser'],
    entry_points={
        'console_scripts': [
            # standalone parsing only supported in v0.4.2 and below (for now)
            'amr-parse = transition_amr_parser.parse:main',
            'amr-machine = transition_amr_parser.amr_machine:main',
        ]
    },
    packages=find_packages(),
    install_requires=install_requires,
)
