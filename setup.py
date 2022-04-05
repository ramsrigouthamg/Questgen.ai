from distutils.core import setup

setup(name='Questgen',
      version='1.0.0',
      description='Question generator from any text',
      author='Questgen contributors',
      author_email='vaibhavtiwarifu@gmail.com',
      packages=['Questgen', 'Questgen.encoding', 'Questgen.mcq'],
      url="https://github.com/ramsrigouthamg/Questgen.ai",
      install_requires=[
         
           'torch==1.8.1',
           'transformers==3.0.2',
           'sense2vec==2.0.0',
           'spacy==3.0.0',
           'strsim==0.0.3',
           'flashtext==2.7',
           'six==1.15.0',
           'networkx==2.4.0',
           'numpy==1.21.0',
           'scipy==1.4.1',
           'scikit-learn==0.22.1',
           'unidecode==1.1.1',
           'future==0.18.2',
           'joblib==0.14.1',
           'python-dateutil==2.8.1',
           'pandas==1.1.1'
      ],
      package_data={'Questgen': ['questgen.py', 'mcq.py', 'train_gpu.py', 'encoding.py']}
      )
