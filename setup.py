from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='model_one',
    version='0.1.7',
    description='beyondml',
    long_description_content_type="text/markdown",
    long_description=README,
    license='MIT',
    packages=find_packages(),
    author='BeyondML',
    author_email='drewxa@beyond-ml.ai',
    keywords=['beyondml', 'model-one', 'gpt-3', 'nlp'],
    url='https://github.com/beyondml/model-one-py',
    download_url='https://pypi.org/project/beyond.ml/'
)

python_requires = ">=3.7, <4"

install_requires = [
    'requests>2.10'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
