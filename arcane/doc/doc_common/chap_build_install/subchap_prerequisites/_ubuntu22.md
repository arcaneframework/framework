# Ubuntu 22.04 {#arcanedoc_build_install_prerequisites_ubuntu22}

[TOC]

## Installation of necessary packages

On Ubuntu 22.04, the versions of CMake and '.Net' are recent enough to be
installed via system packages.

The following commands install the dependencies required for %Arcane (as well as
the optional dependencies `HDF5`and `ParMetis`):

~~~{sh}
sudo apt update
sudo apt install apt-utils build-essential iputils-ping python3 \
  git gfortran libglib2.0-dev libxml2-dev libhdf5-openmpi-dev \
  libparmetis-dev libunwind-dev dotnet6 cmake
~~~

\note If TBB is installed via `apt`, it is version 2020, which is no longer
compatible with %Arcane.
It is therefore necessary either to install a more recent version of TBB (2021+)
or to compile %Arcane with the option:
```sh
 -DARCCORE_ENABLE_TBB=FALSE
```

## Installation of optional packages

~~~{sh}
# For google test:
sudo apt install googletest

# For Ninja:
sudo apt install ninja-build

# For the C# wrapper:
sudo apt install swig4.0

# For Hypre
sudo apt install libhypre-dev

# For PETSc
sudo apt install libpetsc-real-dev

# For Trilinos
sudo apt install libtrilinos-teuchos-dev libtrilinos-epetra-dev \
  libtrilinos-tpetra-dev libtrilinos-kokkos-dev libtrilinos-ifpack2-dev \
  libtrilinos-ifpack-dev libtrilinos-amesos-dev libtrilinos-galeri-dev \
  libtrilinos-xpetra-dev libtrilinos-epetraext-dev \
  libtrilinos-triutils-dev libtrilinos-thyra-dev \
  libtrilinos-kokkos-kernels-dev libtrilinos-rtop-dev \
  libtrilinos-isorropia-dev libtrilinos-belos-dev \

# For Zoltan
sudo apt install libtrilinos-ifpack-dev libtrilinos-anasazi-dev \
  libtrilinos-amesos2-dev libtrilinos-shards-dev libtrilinos-muelu-dev \
  libtrilinos-intrepid2-dev libtrilinos-teko-dev libtrilinos-sacado-dev \
  libtrilinos-stratimikos-dev libtrilinos-shylu-dev \
  libtrilinos-zoltan-dev libtrilinos-zoltan2-dev
~~~

## CUDA

For CUDA support, at least version 3.26 of CMake is required.

\snippet{doc} _cmake.md snippet_build_install_prerequisites_cmake

You can also install CMake directly via [snap](https://snapcraft.io/):
~~~{sh}
sudo snap install --classic cmake
~~~

Next, to install CUDA:
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
