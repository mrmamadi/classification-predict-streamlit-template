from setuptools import setup, find_packages

setup(
    name='en2_edsa_climate',
    version='1.0',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='EDSA classification predict',
    long_description=open('README.md').read(),
    install_requires=['numpy', 'pandas', 'matplotlib', 'seaborn', 'nltk', 'textblob', 'wordcloud', 'streamlit',],
    url='https://github.com/mrmamadi/classification-predict-streamlit-template',
    author=['BNkosi', 'mrmamadi', 'titusndondo', 'MELVA', 'STANLEY', 'ZANELE'],
    author_email=['ebrahim@explore-ai.net', 'bulelaninkosi9@gmail.com', 'mrmamadi@outlook.com', 'tbndondo@gmail.com', 'melva.rirhandzu@gmail.com', 'stanleymachuenek@gmail.com', 'zanelegwamanda99@gmail.com']
)