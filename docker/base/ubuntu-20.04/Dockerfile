# Ubuntu 20.04 dependencies for Alien
FROM ubuntu:20.04

LABEL maintener.email=<cedric.chevalier@cea.fr>

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Europe/Paris

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    cmake \
    ccache \
    gcc g++ \
    ninja-build \
    curl \
    && apt-get clean

## Install linear algebra libraries for alien.
RUN apt-get install --no-install-recommends -y \
    petsc-dev \
    libhypre-dev \
    libsuperlu-dist-dev \
    libboost-dev \
    libboost-program-options-dev \
    libgtest-dev \
    libglib2.0-dev \
    libxml2-dev \
    libhdf5-openmpi-dev \
    doxygen \
    python3-breathe sphinx-rtd-theme-common \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# gtest package contains only src, we need to compile it
RUN export BDIR=$(mktemp -d) && cd ${BDIR} && \
    cmake /usr/src/googletest && \
    cmake --build . --target install && \
    rm -rf ${BDIR}

# Install Arccon and Arccore
RUN export BDIR=$(mktemp -d) && cd ${BDIR} && \
    curl -L https://gitlab.com/cea-ifpen/arccon/-/archive/v1.2.0/arccon-v1.2.0.tar.gz | tar xzf - --strip-components=1 && \
    mkdir build_dir && cmake -B build_dir -DCMAKE_INSTALL_PREFIX=/opt/arccon && \
    cmake --build build_dir && cmake --install build_dir && \
    rm -rf ${BDIR}

RUN export BDIR=$(mktemp -d) && cd ${BDIR} && \
    curl -L https://gitlab.com/cea-ifpen/arccore/-/archive/v2.0.1/arccore-v2.0.1.tar.gz | tar xzf - --strip-components=1 && \
    mkdir build_dir && \
    cmake -B build_dir -DCMAKE_PREFIX_PATH=/opt/arccon -DCMAKE_INSTALL_PREFIX=/opt/arccore -DBUILD_SHARED_LIBS=ON && \
    cmake --build build_dir && cmake --install build_dir && \
    rm -rf ${BDIR}
