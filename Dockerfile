FROM ubuntu:18.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-setuptools python3-dev git g++ build-essential cmake pkg-config wget

WORKDIR /src/jupyter

# Install all Python packages
RUN pip3 install setuptools_rust Cython
RUN pip3 install git+https://github.com/ramsrigouthamg/Questgen.ai
RUN pip3 install git+https://github.com/boudinfl/pke.git
RUN python3 -m nltk.downloader universal_tagset
RUN python3 -m spacy download en

# Download and unpack sense2vec model
RUN wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
RUN tar -xvf  s2v_reddit_2015_md.tar.gz

# Install Jupyter notebook
RUN pip3 install jupyter

# Download and set `tini` as entrypoint (see https://github.com/krallin/tini for reasons)
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# Execute Jupyter notebook
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]