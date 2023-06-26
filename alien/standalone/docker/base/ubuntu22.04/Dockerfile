# Ubuntu 22.04 dependencies for Alien
FROM ubuntu:22.04

LABEL maintainer.email=<cedric.chevalier@cea.fr>

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Europe/Paris

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    cmake \
    ccache \
    gcc g++ gdb \
    ninja-build \
    curl \
    git \
    && apt-get clean

## Install linear algebra libraries for alien.
RUN apt-get install --no-install-recommends -y \
    petsc-dev \
    libhypre-dev \
    libsuperlu-dist-dev \
    trilinos-all-dev \
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

# Install and build Ginkgo
# git must be installed when configuring to download external products.
RUN export BDIR=$(mktemp -d) && cd ${BDIR} && \
    mkdir -p /usr/src/ginkgo && \
    curl -L https://github.com/ginkgo-project/ginkgo/archive/refs/tags/v1.4.0.tar.gz | tar xz -C /usr/src/ginkgo --strip-components=1 && \
    cmake -DGINKGO_BUILD_TESTS:BOOL=OFF -DGINKGO_BUILD_EXAMPLES:BOOL=OFF -DGINKGO_BUILD_BENCHMARKS:BOOL=OFF /usr/src/ginkgo -G Ninja && \
    cmake --build . --target install && \
    rm -rf ${BDIR}

