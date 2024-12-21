# Ubuntu 24.04 {#arcanedoc_build_install_prerequisites_ubuntu24}

[TOC]

## Installation des packages nécessaires

Sur Ubuntu 24.04, la version de CMake et de '.Net' sont suffisamment
récentes pour pouvoir être installés via les packages système.

Les commandes suivantes permettent d'installer les dépendances
nécessaires pour %Arcane (ainsi que les dépendances optionnelles `HDF5` et `ParMetis`):

~~~{sh}
sudo apt update
sudo apt install -y apt-utils build-essential iputils-ping python3 \
  git gfortran libglib2.0-dev libxml2-dev libhdf5-openmpi-dev \
  libparmetis-dev libunwind-dev dotnet8 cmake
~~~

Pour compiler Alien en plus d'%Arcane, il est nécessaire d'installer un package en plus :
~~~{sh}
sudo apt install -y libboost-program-options-dev
~~~

## Installation des packages optionnels

~~~{sh}
# Pour google test:
sudo apt install -y googletest

# Pour Ninja:
sudo apt install -y ninja-build

# Pour Hypre
sudo apt install -y libhypre-dev

# Pour PETSc
sudo apt install -y libpetsc-real-dev

# Pour Trilinos
sudo apt install -y libtrilinos-teuchos-dev libtrilinos-epetra-dev \
  libtrilinos-tpetra-dev libtrilinos-kokkos-dev libtrilinos-ifpack2-dev \
  libtrilinos-ifpack-dev libtrilinos-amesos-dev libtrilinos-galeri-dev \
  libtrilinos-xpetra-dev libtrilinos-epetraext-dev \
  libtrilinos-triutils-dev libtrilinos-thyra-dev \
  libtrilinos-kokkos-kernels-dev libtrilinos-rtop-dev \
  libtrilinos-isorropia-dev libtrilinos-belos-dev

# Pour Zoltan
sudo apt install -y libtrilinos-ifpack-dev libtrilinos-anasazi-dev \
  libtrilinos-amesos2-dev libtrilinos-shards-dev libtrilinos-muelu-dev \
  libtrilinos-intrepid2-dev libtrilinos-teko-dev libtrilinos-sacado-dev \
  libtrilinos-stratimikos-dev libtrilinos-shylu-dev \
  libtrilinos-zoltan-dev libtrilinos-zoltan2-dev

# Pour le wrapper C#:
sudo apt install -y swig
~~~

## CUDA

Aujourd'hui, pour utiliser CUDA sur Ubuntu 24.04, il faut utiliser GCC 12 ou moins.

Il est donc nécessaire d'installer GCC 12 et de compiler %Arcane avec GCC 12.

Donc, pour installer GCC 12 et CUDA sur Ubuntu 24.04 :

~~~{sh}
sudo apt update
sudo apt install g++-12 gcc-12 nvidia-cuda-toolkit
~~~

Ensuite, pour compiler %Arcane, lors de la configuration CMake, il faudra spécifier,
en plus des options pour CUDA, la bonne version de GCC :

~~~{sh}
cmake -S /path/to/sources -B /path/to/build \
-DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_C_COMPILER=gcc-12 \
-DARCANE_ACCELERATOR_MODE=CUDANVCC \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11/bin/nvcc \
-DARCCORE_CXX_STANDARD=20
~~~

Voir la page suivante pour plus d'informations sur la compilation.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_build_install_prerequisites
</span>
<span class="next_section_button">
\ref arcanedoc_build_install_build
</span>
</div>
