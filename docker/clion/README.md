# Development with CLion and containers

## Building the container

```shell
podman build --build-arg UID=<uid> --build-arg GID=<gid> . -t alien-ubuntu2004
```

## Running the container

```shell
podman run -d --image-volume=tmpfs --cap-add sys_ptrace -p 127.0.0.1:2004:22 alien-ubuntu2004
```