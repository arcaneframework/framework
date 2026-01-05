# Ubuntu 22.04 {#arcanedoc_build_install_prerequisites_ubuntu22}

[TOC]

## Installation des packages nécessaires

Sur Ubuntu 22.04, les versions de CMake et de '.Net' sont suffisamment
récentes pour pouvoir être installés via les packages système.

Les commandes suivantes permettent d'installer les dépendances
nécessaires pour %Arcane (ainsi que les dépendances optionnelles `HDF5` et `ParMetis`):

~~~{sh}
sudo apt update
sudo apt install apt-utils build-essential iputils-ping python3 \
  git gfortran libglib2.0-dev libxml2-dev libhdf5-openmpi-dev \
  libparmetis-dev libunwind-dev dotnet6 cmake
~~~

\note Si TBB est installé via `apt`, il s'agit de la version 2020 qui n'est plus compatible avec %Arcane.
Il est donc nécessaire, soit d'installer une version plus récente de TBB (2021+), soit de compiler %Arcane avec
l'option :
```sh
 -DARCCORE_ENABLE_TBB=FALSE
```

## Installation des packages optionnels

~~~{sh}
# Pour google test:
sudo apt install googletest

# Pour Ninja:
sudo apt install ninja-build

# Pour le wrapper C#:
sudo apt install swig4.0

# Pour Hypre
sudo apt install libhypre-dev

# Pour PETSc
sudo apt install libpetsc-real-dev

# Pour Trilinos
sudo apt install libtrilinos-teuchos-dev libtrilinos-epetra-dev \
  libtrilinos-tpetra-dev libtrilinos-kokkos-dev libtrilinos-ifpack2-dev \
  libtrilinos-ifpack-dev libtrilinos-amesos-dev libtrilinos-galeri-dev \
  libtrilinos-xpetra-dev libtrilinos-epetraext-dev \
  libtrilinos-triutils-dev libtrilinos-thyra-dev \
  libtrilinos-kokkos-kernels-dev libtrilinos-rtop-dev \
  libtrilinos-isorropia-dev libtrilinos-belos-dev \

# Pour Zoltan
sudo apt install libtrilinos-ifpack-dev libtrilinos-anasazi-dev \
  libtrilinos-amesos2-dev libtrilinos-shards-dev libtrilinos-muelu-dev \
  libtrilinos-intrepid2-dev libtrilinos-teko-dev libtrilinos-sacado-dev \
  libtrilinos-stratimikos-dev libtrilinos-shylu-dev \
  libtrilinos-zoltan-dev libtrilinos-zoltan2-dev
~~~

## CUDA

Pour le support de CUDA, il faut au moins la version 3.26 de CMake.

\snippet{doc} _cmake.md snippet_build_install_prerequisites_cmake

Vous pouvez aussi installer directement CMake via [snap](https://snapcraft.io/):
~~~{sh}
sudo snap install --classic cmake
~~~

Ensuite, pour installer CUDA :
~~~{sh}
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-3
~~~




____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_build_install_prerequisites
</span>
<span class="next_section_button">
\ref arcanedoc_build_install_build
</span>
</div>
