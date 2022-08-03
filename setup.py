from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup_args = dict(
    name='tune_the_model',
    version='0.1.19',
    description='beyondml',
    long_description_content_type="text/markdown",
    long_description=README,
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    author='BeyondML',
    author_email='pavel.gavrilov@beyond.ml',
    keywords=['beyondml', 'tune-the-model', 'gpt-3', 'nlp'],
    url='https://github.com/tune-the-model/tune-the-model-py',
    download_url='https://pypi.org/project/beyond.ml/'
)

python_requires = ">=3.7, <4"

if __name__ == '__main__':
    setup(**setup_args, install_requires=requirements)
