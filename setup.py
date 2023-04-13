from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="plug_ai",
    version="23.03a",
    license='MIT',
    author='C.Frindel & M.Sdika',
    packages=find_packages(),
    url='https://github.com/CaroleFri/Plug-AI',
    keywords='plug_ai medical ai framework pytorch segmentation nifti',
    install_requires=requirements,
)

setup(
    name='example_publish_pypi_medium',
    version='0.6',
    license='MIT',
    author="Giorgos Myrianthous",
    author_email='email@example.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/gmyrianthous/example-publish-pypi',
    keywords='example project',
    install_requires=[
          'scikit-learn',
      ],

)