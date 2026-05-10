# Ubuntu 26.04 (Resolute Raccoon) {#arcanedoc_build_install_prerequisites_ubuntu26}

[TOC]

## Installation des packages nécessaires

Sur Ubuntu 26.04, les versions des dépendances nécessaires pour
compiler %Arcane (GCC, CMake, '.Net', ...) sont suffisament récentes
pour pouvoir être installés via les packages système.

Les commandes suivantes permettent d'installer les dépendances
nécessaires pour %Arcane

~~~{sh}
sudo apt update
sudo apt install -y apt-utils build-essential iputils-ping python3 \
  git gfortran libglib2.0-dev libxml2-dev \
  libunwind-dev software-properties-common dotnet-sdk-10.0 cmake
~~~

Pour compiler Alien en plus d'%Arcane, il est nécessaire d'installer un package en plus :
~~~{sh}
sudo apt install -y libboost-program-options-dev
~~~

## Installation des packages optionnels

\note Actuellement (avril 2026) il n'y a pas encore de package
dans Ubuntu 26.04 pour ParMetis.

~~~{sh}
# Pour HDF5
sudo apt libhdf5-openmpi-dev

# Pour google test:
sudo apt install -y googletest

# Pour Ninja:
sudo apt install -y ninja-build

# Pour Hypre
sudo apt install -y libhypre-dev

# Pour PETSc
sudo apt install -y libpetsc-real-dev

# Pour le wrapper C#:
sudo apt install -y swig
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
