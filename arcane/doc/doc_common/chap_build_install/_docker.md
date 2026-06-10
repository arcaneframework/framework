# Docker images {#arcanedoc_build_install_docker}

[TOC]

## Images including the Arcane Framework

These images contain everything necessary to work on an application using the
%Arcane framework. %Arcane and all necessary packages are installed in them.

The GitHub repository hosting these images is
[here](https://github.com/arcaneframework/containers). This repository contains
a complete Readme explaining how the image tags are formed.

The DockerFiles are generated on demand and some images are updated weekly.

The images can be
found [here](https://github.com/arcaneframework/containers/pkgs/container/arcane_ubuntu-2404).

Usage example:
```sh
IMAGE_ID=$(docker pull ghcr.io/arcaneframework/arcane_ubuntu-2404:gcc-14_full_release_latest)
CONTAINER_ID=$(docker run -dt "$IMAGE_ID")
docker exec -it "$CONTAINER_ID" bash
```

Or with [Distrobox](https://github.com/89luca89/distrobox):
```sh
distrobox create --init \
  --name U24FullArcaneRelease \
  --image ghcr.io/arcaneframework/arcane_ubuntu-2404:gcc-14_full_release_latest \
  --additional-packages vim

distrobox enter U24FullArcaneRelease
```

## Images not including the Arcane Framework

These images were originally intended for the framework repository's CI.
However, these images can also be used for development since they contain all
the necessary packages for %Arcane.

The GitHub repository hosting these images is
[here](https://github.com/arcaneframework/framework-ci). This repository
contains a complete Readme explaining how the image tags are formed. Each image
has a dedicated branch.

The images can be found
[here](https://github.com/arcaneframework/framework-ci/pkgs/container/ubuntu-2404).

Usage example:
```sh
IMAGE_ID=$(docker pull ghcr.io/arcaneframework/ubuntu-2404:full_stable)
CONTAINER_ID=$(docker run -dt "$IMAGE_ID")
docker exec -it "$CONTAINER_ID" bash
```

Or with [Distrobox](https://github.com/89luca89/distrobox):
```sh
distrobox create --init \
  --name U24Full \
  --image ghcr.io/arcaneframework/ubuntu-2404:full_stable \
  --additional-packages vim

distrobox enter U24Full
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_build_install
</span>
</div>
