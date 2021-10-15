# Ubuntu 20.04 dependencies for Alien
FROM ubuntu:20.04

LABEL maintainer.email=<cedric.chevalier@cea.fr>

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
