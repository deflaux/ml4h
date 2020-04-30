ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root

## Install system dependencies needed
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
     wget \
     unzip \
     build-essential \
     libcurl4-gnutls-dev \
     libxml2-dev \
     libssl-dev
#     automake \
#     ## This section installs libraries
#     libnetcdf-dev \
#     libhdf5-serial-dev \
#     libfftw3-dev \
#     libopenbabel-dev \
#     libopenmpi-dev \
#     libexempi3 \
#     libgdal-dev \
#     libcairo2-dev \
#     libtiff5-dev \
#     libgsl0-dev \
#     libgtk2.0-dev \
#     libgl1-mesa-dev \
#     libglu1-mesa-dev \
#     libgmp3-dev \
#     libhdf5-dev \
#     libncurses-dev \
#     libxpm-dev \
#     libv8-3.14-dev \
#     libgtkmm-2.4-dev \
#     libmpfr-dev \
#     libudunits2-dev \
#     libmodule-build-perl \
#     libapparmor-dev \
#     libgeos-dev \
#     libprotoc-dev \
#     librdf0-dev \
#     libmagick++-dev \
#     libsasl2-dev \
#     libpoppler-cpp-dev \
#     libprotobuf-dev \
#     libpq-dev \
#     libperl-dev \
#     ## software - perl extentions and modules
#     libarchive-extract-perl \
#     libfile-copy-recursive-perl \
#     libcgi-pm-perl \
#     libdbi-perl \
#     libdbd-mysql-perl \
#     libxml-simple-perl \
#     ## Databases and other software
#     sqlite \
#     openmpi-bin \
#     mpi-default-bin \
#     openmpi-common \
#     openmpi-doc \
#     tcl8.5-dev \
#     imagemagick \
#     tabix \
#     ggobi \
#     graphviz \
#     protobuf-compiler \
#     jags \
#     ## Additional resources
#     xfonts-100dpi \
#     xfonts-75dpi \
#     biber \

# # Install libsbml
# RUN cd /tmp \
#     && curl -O https://s3.amazonaws.com/linux-provisioning/libSBML-5.10.2-core-src.tar.gz \
#     && tar zxf libSBML-5.10.2-core-src.tar.gz \
#     && cd libsbml-5.10.2 \
#     && ./configure --enable-layout \
#     && make \
#     && make install \
#     && ldconfig \
#     && rm -rf /tmp/libsbml-5.10.2 \
#     && rm -rf /tmp/libSBML-5.10.2-core-src.tar.gz

# ## Pip runs as root
# ENV PIP_USER=false

# ## Install python packages needed for a few Bioc packages
# RUN apt-get update \
#     && apt-get install -yq --no-install-recommends python3.7-dev \
#     && pip3 -V \
#     && pip3 install --upgrade pip \
#     && pip3 install cwltool==1.0.20190228155703 \
#     && pip3 install pandas==0.25.3 \
#     && pip3 install pyyaml \
#     && pip3 install scikit-learn==0.21.3 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get -y install sudo \
    libblas-dev \
    liblapack-dev \
    gfortran

RUN apt-get -y install software-properties-common && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/' && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    r-base 
    
## Create jupyter-user
RUN useradd -ms /bin/bash jupyter-user && \
    echo "jupyter-user:jupyter-user" | chpasswd && \
    adduser jupyter-user sudo
USER jupyter-user
## pip runs as jupyter-user
# ENV PIP_USER=true

# Take care of plink
RUN mkdir /home/jupyter-user/packages && \
    mkdir /home/jupyter-user/src && \
    wget http://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20200121.zip -O /home/jupyter-user/packages/plink.zip && \
    cd /home/jupyter-user/packages && \
    unzip /home/jupyter-user/packages/plink.zip

RUN wget http://keinanlab.cb.bscb.cornell.edu/data/xwas/XWAS_v3.0.tar.gz -O /home/jupyter-user/packages/XWAS.tar.gz && \
    cd /home/jupyter-user/packages && \
    tar xzvf /home/jupyter-user/packages/XWAS.tar.gz

RUN wget https://cran.r-project.org/src/contrib/Archive/GenABEL/GenABEL_1.8-0.tar.gz -O /home/jupyter-user/src/GenABEL.tar.gz && \
    wget https://cran.r-project.org/src/contrib/Archive/GenABEL.data/GenABEL.data_1.0.0.tar.gz -O /home/jupyter-user/src/GenABEL.data.tar.gz

ADD install.R /home/jupyter-user/src/
#ENV R_LIBS_USER /home/jupyter-user/packages/R/lib
USER root
#RUN mkdir -p /home/jupyter-user/packages/R/lib && \
#    R -f /home/jupyter-user/src/install.R jupyter-user

RUN apt install -y libzmq3-dev libcurl4-openssl-dev libssl-dev
RUN pip install jupyter && \
    pip install --upgrade jupyter_client
    
RUN  R -f /home/jupyter-user/src/install.R
  
# ## Install Bioconductor Core packages
# RUN curl -O https://raw.githubusercontent.com/Bioconductor/anvil-docker/master/anvil-rstudio-bioconductor/install.R \
# 	&& R -f install.R $USER \
# 	&& rm -rf install.R

RUN mkdir -p /home/jupyter-user/notebooks
#ADD jupyter.R /home/jupyter-user/src
#RUN R -f /home/jupyter-user/src/jupyter.R 


