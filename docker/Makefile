# Dockerconfig
UID=$(shell id --user)
GID=$(shell id --group)
DOCKER_CMD=podman

IMAGE_BASE_NAME=alien-base
IMAGE_VERSION=ubuntu20.04
REPO=ghcr.io/arcaneframework

.phony: build push

.DEFAULT_GOAL := build

build: base/ubuntu-20.04/Dockerfile
	${DOCKER_CMD} build -t ${REPO}/${IMAGE_BASE_NAME}:${IMAGE_VERSION} base/ubuntu-20.04

push: build
	${DOCKER_CMD} push  ${REPO}/${IMAGE_BASE_NAME}:${IMAGE_VERSION}
