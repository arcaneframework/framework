# Ubuntu 24.04 {#arcanedoc_build_install_prerequisites_ubuntu24}

[TOC]

## Installation of necessary packages

On Ubuntu 24.04, the versions of the dependencies required to compile %Arcane
(GCC, CMake, '.Net', ...) are recent enough to be installed via system packages.

The following commands allow you to install the dependencies required for
%Arcane (as well as the optional dependencies `HDF5` and `ParMetis`):

~~~{sh}
sudo apt update
sudo apt install -y apt-utils build-essential iputils-ping python3 \
  git gfortran libglib2.0-dev libxml2-dev libhdf5-openmpi-dev \
  libparmetis-dev libunwind-dev dotnet8 cmake
~~~

To compile Alien in addition to %Arcane, it is necessary to install one
additional package:
~~~{sh}
sudo apt install -y libboost-program-options-dev
~~~

## Installation of optional packages

~~~{sh}
# For google test:
sudo apt install -y googletest

# For Ninja:
sudo apt install -y ninja-build

# For Hypre
sudo apt install -y libhypre-dev

# For PETSc
sudo apt install -y libpetsc-real-dev

# For Trilinos
sudo apt install -y libtrilinos-teuchos-dev libtrilinos-epetra-dev \
  libtrilinos-tpetra-dev libtrilinos-kokkos-dev libtrilinos-ifpack2-dev \
  libtrilinos-ifpack-dev libtrilinos-amesos-dev libtrilinos-galeri-dev \
  libtrilinos-xpetra-dev libtrilinos-epetraext-dev \
  libtrilinos-triutils-dev libtrilinos-thyra-dev \
  libtrilinos-kokkos-kernels-dev libtrilinos-rtop-dev \
  libtrilinos-isorropia-dev libtrilinos-belos-dev

# For Zoltan
sudo apt install -y libtrilinos-ifpack-dev libtrilinos-anasazi-dev \
  libtrilinos-amesos2-dev libtrilinos-shards-dev libtrilinos-muelu-dev \
  libtrilinos-intrepid2-dev libtrilinos-teko-dev libtrilinos-sacado-dev \
  libtrilinos-stratimikos-dev libtrilinos-shylu-dev \
  libtrilinos-zoltan-dev libtrilinos-zoltan2-dev

# For the C# wrapper:
sudo apt install -y swig
~~~

## CUDA

Currently, to use CUDA on Ubuntu 24.04, you must use GCC 12 or less.

It is therefore necessary to install GCC 12 and compile %Arcane with GCC 12.

Thus, to install GCC 12 and CUDA on Ubuntu 24.04:

~~~{sh}
sudo apt update
sudo apt install g++-12 gcc-12 nvidia-cuda-toolkit
~~~

Next, to compile %Arcane, when configuring CMake, you will need to specify, in
addition to the CUDA options, the correct GCC version:

~~~{sh}
cmake -S /path/to/sources -B /path/to/build \
-DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_C_COMPILER=gcc-12 \
-DARCANE_ACCELERATOR_MODE=CUDANVCC \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11/bin/nvcc \
-DARCCORE_CXX_STANDARD=20
~~~

See the following page for more information on compilation.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_build_install_prerequisites
</span>
<span class="next_section_button">
\ref arcanedoc_build_install_build
</span>
</div>
