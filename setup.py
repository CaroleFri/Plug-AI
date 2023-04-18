from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="plug_ai",
    version="23.03a",
    license='Cecill-B',
    author='C.Frindel & M.Sdika',
    packages=find_packages(),
    url='https://github.com/CaroleFri/Plug-AI',
    keywords='plug_ai medical ai framework pytorch segmentation nifti',
    install_requires=requirements,
    package_data={
        'plug_ai': ['ressources/default_config.yaml',
                    'ressources/*'
                   ],
    },
)
