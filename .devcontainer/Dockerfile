FROM ghcr.io/arcaneframework/alien-base:ubuntu20.04

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install --no-install-recommends -y git nodejs ninja-build ccache gdb valgrind\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
