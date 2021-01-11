from setuptools import setup, find_packages

VERSION = '0.4.0'

# this is what usually goes on requirements.txt
install_requires = [
    # NOTE: For PPCs we need to relax these two to 1.2 and 2.0.16
    'torch<=1.2,<=1.3',
    'spacy<=2.0.16,<=2.2.3',
    #
    'fairseq==0.8.0',
    'h5py',
    'tqdm',
    # for scoring
    'smatch==1.0.4',
    # for debugging
    'ipdb',
    'line_profiler',
    'pyinstrument',
    'packaging'
    # these may be missing
    # dataclasses hydra-core omegaconf
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
