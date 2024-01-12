# Images Docker {#arcanedoc_build_install_docker}

[TOC]

## Les images incluant le framework Arcane

Ces images contiennent tout ce qui est nécessaire pour travailler sur une application
utilisant le framework %Arcane. %Arcane et tous les packages nécessaires y sont installés.

Le dépôt GitHub hébergeant ces images est [ici](https://github.com/arcaneframework/containers).
Ce dépôt contient un Readme complet expliquant comment les tags des images sont formés.

Les DockerFiles sont générés à la demande et certaines images sont mises à jour toutes les semaines.

Les images peuvent être trouvées [ici](https://github.com/arcaneframework/containers/pkgs/container/arcane_ubuntu-2204).

Exemple d'utilisation :
```shell
IMAGE_ID=$(docker pull ghcr.io/arcaneframework/arcane_ubuntu-2204:gcc-12_full_release_latest)
CONTAINER_ID=$(docker run -dt "$IMAGE_ID")
docker exec -it "$CONTAINER_ID" bash
```

## Les images n'incluant pas le framework Arcane

Ces images était destinées à l'origine aux CI du dépôt framework. Mais ces images peuvent
être aussi utilisées pour le développement étant donné qu'elles contiennent tous les packages
nécessaires pour %Arcane.

Le dépôt GitHub hébergeant ces images est [ici](https://github.com/arcaneframework/framework-ci).
Ce dépôt contient un Readme complet expliquant comment les tags des images sont formés.
Chaque image a une branche qui lui est dédiée.

Les images peuvent être trouvées [ici](https://github.com/arcaneframework/framework-ci/pkgs/container/ubuntu-2204).

Exemple d'utilisation :
```shell
IMAGE_ID=$(docker pull ghcr.io/arcaneframework/ubuntu-2204:gcc-12_full_latest)
CONTAINER_ID=$(docker run -dt "$IMAGE_ID")
docker exec -it "$CONTAINER_ID" bash
```


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_build_install
</span>
</div>
