from setuptools import setup, find_packages

VERSION = '0.5.3'

install_requires = [
    "torch==1.13.1",
    'numpy<=1.23.5',
    'ipython<=8.12', # python 3.8 vs ipython 8.13 incompatibility
    'tqdm>=4.55.0',
    'fairseq==0.10.2', # seems to not work on python 3.9 for some platforms
    'packaging>=20.8',
    'requests>=2.25.1',
    # for data (ELMO embeddings)
    'h5py>=3.0.0',
    'python-dateutil>=2.8.1',
    # for scoringy
    'penman>=1.1.0',
    # needs tools to be importable > 1.0.4
    'smatch>=1.0.3.2',
    # for debugging
    'ipdb',
    'line_profiler>=4.0.2',
    'pyinstrument>=4.4.0',
    # for aws download
    'boto3>=1.26.1',
    'progressbar',
]

if __name__ == '__main__':
    setup(
        name='transition_amr_parser',
        version=VERSION,
        description="Trasition-based neural parser",
        package_dir={"": "src"},
        # packages=['fairseq_ext', 'transition_amr_parser'],
        # packages=['neural_parser'],
        packages=find_packages("src", exclude=('cenv_*', 'configs', 'tests', 'DATA','dist','docker','run','scripts','service','*egg-info')),
        package_data={'': ['*.txt', '*.md', '*.opt', '*.cu', '*.cpp']},
        entry_points={
            'console_scripts': [
                'amr-parse = transition_amr_parser.parse:main',
                'amr-machine = transition_amr_parser.amr_machine:main',
            ]
        },
        py_modules=['fairseq_ext', 'transition_amr_parser',"ibm_neural_aligner"],
        install_requires=install_requires,
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Natural Language :: English",
        ],
    )

