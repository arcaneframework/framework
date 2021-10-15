# Dockerconfig
UID=$(shell id --user)
GID=$(shell id --group)
DOCKER_CMD=podman

IMAGE_BASE_NAME=alien-base
IMAGE_VERSION=ubuntu20.04
REPO=ghcr.io/arcaneframework

.phony: run build_all push build_base build_clion

.DEFAULT_GOAL := run

build_all: build_clion build_base

build_clion: build_base clion/Dockerfile
	${DOCKER_CMD} build --build-arg UID=${UID} --build-arg GID=${GID} -t ${REPO}/clion-${IMAGE_BASE_NAME}:${IMAGE_VERSION} clion

build_base: base/ubuntu-20.04/Dockerfile
	${DOCKER_CMD} build -t ${REPO}/${IMAGE_BASE_NAME}:${IMAGE_VERSION} base/ubuntu-20.04

run: build_clion
	${DOCKER_CMD} run --rm -d --image-volume=tmpfs --cap-add sys_ptrace -p 127.0.0.1:2004:22 ${REPO}/clion-${IMAGE_BASE_NAME}:${IMAGE_VERSION}

push: build_base
	${DOCKER_CMD} push  ${REPO}/${IMAGE_BASE_NAME}:${IMAGE_VERSION}
