# Ubuntu 26.04 (Resolute Raccoon) {#arcanedoc_build_install_prerequisites_ubuntu26}

[TOC]

## Installation of necessary packages

On Ubuntu 26.04, the versions of dependencies required to compile %Arcane (GCC,
CMake, '.Net', ...) are sufficiently recent to be installed via system packages.

The following commands allow you to install the dependencies required for
%Arcane:

~~~{sh}
sudo apt update
sudo apt install -y apt-utils build-essential iputils-ping python3 \
  git gfortran libglib2.0-dev libxml2-dev \
  libunwind-dev software-properties-common dotnet-sdk-10.0 cmake
~~~

To compile Alien in addition to %Arcane, it is necessary to install one
additional package:
~~~{sh}
sudo apt install -y libboost-program-options-dev
~~~

## Installation of optional packages

\note Currently (April 2026), there is no package in Ubuntu 26.04 for ParMetis
yet.

~~~{sh}
# For HDF5
sudo apt libhdf5-openmpi-dev

# For googletest:
sudo apt install -y googletest

# For Ninja:
sudo apt install -y ninja-build

# For Hypre
sudo apt install -y libhypre-dev

# For PETSc
sudo apt install -y libpetsc-real-dev

# For the C# wrapper:
sudo apt install -y swig
~~~

See the next page for more information on compilation.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_build_install_prerequisites
</span>
<span class="next_section_button">
\ref arcanedoc_build_install_build
</span>
</div>
