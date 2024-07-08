# Ubuntu 20.04 {#arcanedoc_build_install_prerequisites_ubuntu20}

[TOC]

## Installation des packages nécessaires

Les commandes suivantes permettent d'installer les dépendances
nécessaires pour %Arcane (ainsi que les dépendances optionnelles `HDF5` et `ParMetis`):

~~~{sh}
sudo apt update
sudo apt install apt-utils build-essential iputils-ping python3 \
git gfortran libglib2.0-dev libxml2-dev libhdf5-openmpi-dev \
libparmetis-dev wget gcc-11 g++-11

wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install -y apt-transport-https dotnet-sdk-6.0
~~~

## CUDA

Pour le support de CUDA, il faut au moins la version 3.26 de CMake.

\snippet{doc} _cmake.md snippet_build_install_prerequisites_cmake

Ensuite, pour installer CUDA :
~~~{sh}
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
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
