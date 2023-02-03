from setuptools import setup, find_packages

VERSION = '0.5.2'

# this is what usually goes on requirements.txt
install_requires = [ ]
# install_requires = [
#     # 'torch==1.10.1',
#     # 'torch==1.11.0+cu113',
#     'torch==1.11.0',
#     # "torch>=1.13",
#     # 'torchvision>=0.12.0',
#     # 'torchaudio>=0.8.0',
#     # 'torch-scatter==2.0.9',
#     'torch-scatter==2.1.0',
#     'tqdm>=4.55.0',
#     'fairseq==0.10.2',
#     'packaging>=20.8',
#     'requests>=2.25.1',
#     # for data (ELMO embeddings)
#     'h5py>=3.0.0',
#     'python-dateutil>=2.8.1',
#     # for scoring
#     'penman>=1.1.0',
#     # needs tools to be importable > 1.0.4
#     'smatch>=1.0.3.2',
#     # for debugging
#     'ipdb>=0.13.0',
#     'line_profiler==4.0.2',
#     'pyinstrument==4.4.0'
# ]

# setup(
#     name='transition_amr_parser',
#     version=VERSION,
#     description="Trasition-based AMR parser",
#     package_dir={"": "src"},
#     # packages=find_packages("src"),
#     # packages = find_packages(where="src"),
#     packages=['fairseq_ext', 'transition_amr_parser'],
    
#     # package_data={"transformers": ["py.typed", "*.cu", "*.cpp", "*.cuh", "*.h", "*.pyx"]},
#     zip_safe=False,
#     # extras_require=extras,
#     # entry_points={"console_scripts": ["transformers-cli=transformers.commands.transformers_cli:main"]},
    
    
#     # py_modules=['fairseq_ext','transition_amr_parser'],
#     entry_points={
#         'console_scripts': [
#             'amr-parse = neural_parser.transition_amr_parser.parse:main',
#             'amr-machine = neural_parser.transition_amr_parser.amr_machine:main',
#         ]
#     },
#     # packages=find_packages(),
#     install_requires=install_requires,
# )
# setup(
#     name='transition_amr_parser',
#     version=VERSION,
#     description="Trasition-based AMR parser",
#     py_modules=['transition_amr_parser'],
#     entry_points={
#         'console_scripts': [
#             'amr-parse = transition_amr_parser.parse:main',
#             'amr-machine = transition_amr_parser.amr_machine:main',
#         ]
#     },
#     packages=find_packages(),
#     install_requires=install_requires,
# )

if __name__ == '__main__':
    setup(
        name='transition_amr_parser',
        version=VERSION,
        description="Trasition-based neural parser",
        # package_dir={"": "src"},
        # packages=['fairseq_ext', 'transition_amr_parser'],
        # packages=['neural_parser'],
        packages=find_packages(exclude=('cenv_*', 'configs', 'tests', 'DATA','dist','docker','ibm_neural_aligner','run','scripts','service','*egg-info')),
        package_data={'': ['*.txt', '*.md', '*.opt', '*.cu', '*.cpp']},
        entry_points={
            'console_scripts': [
                'amr-parse = transition_amr_parser.parse:main',
                'amr-machine = transition_amr_parser.amr_machine:main',
            ]
        },
        py_modules=['neural_parser.fairseq_ext', 'neural_parser.transition_amr_parser'],
        install_requires=install_requires,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Natural Language :: English",
        ],
    )