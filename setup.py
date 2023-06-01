from distutils.core import setup

setup(name='Questgen',
      version='1.0.0',
      description='Question generator from any text',
      author='Questgen contributors',
      author_email='vaibhavtiwarifu@gmail.com',
      packages=['Questgen', 'Questgen.encoding', 'Questgen.mcq'],
      url="https://github.com/ramsrigouthamg/Questgen.ai",
      install_requires=[
         
           'torch',
           'transformers',
           'sense2vec',
           'strsim',
           'six',
           'networkx',
           'numpy',
           'scipy',
           'scikit-learn',
           'unidecode',
           'future',
           'joblib',
           'pytz',
           'python-dateutil',
           'flashtext',
           'pandas'
      ],
      package_data={'Questgen': ['questgen.py', 'mcq.py', 'train_gpu.py', 'encoding.py']}
      )
