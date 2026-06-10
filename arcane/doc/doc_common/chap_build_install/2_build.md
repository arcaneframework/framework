# Compilation {#arcanedoc_build_install_build}

[TOC]

Compilation must be done in a directory different from the one containing the
sources.

## Source retrieval

To retrieve the sources:

~~~{sh}
git clone --recurse-submodules https://github.com/arcaneframework/framework
~~~

or

~~~{sh}
git clone https://github.com/arcaneframework/framework
cd framework && git submodule update --init --recursive
~~~

## Compilation

By default, %Arcane and Alien are compiled if the prerequisites are available.
The CMake variable `ARCANEFRAMEWORK_BUILD_COMPONENTS` contains the list of
repository components to compile. This list can contain the following values:

- `%Arcane`
- `Alien`

By default, the value is `%Arcane;Alien`, and thus both components are compiled.

To compile %Arcane and Alien, you must proceed as follows:

~~~{sh}
mkdir /path/to/build
cmake -S /path/to/sources -B /path/to/build
cmake --build /path/to/build
cmake --build /path/to/build --target install
~~~

By default, installation takes place in `/usr/local` if the
`CMAKE_INSTALL_PREFIX` option is not specified.

<details>
<summary>Available general configuration options</summary>
General options:
<table>
<tr><th>Option <th>Value <th>Description
<tr><td>`CMAKE_INSTALL_PREFIX` <td>`/path/to/install` <td>Choice of an installation directory
<tr><td>`ARCANEFRAMEWORK_BUILD_COMPONENTS` <td>`%Arcane` or `Alien` or `%Arcane;Alien` <td>Component(s) to compile 
<tr><td>`ARCCORE_CXX_STANDARD` <td>`20` (the default), `23` or `26` <td>Choice of C++ standard to use
<tr><td>`ARCANE_ENABLE_TESTS` <td>`ON`/`OFF` <td>Enable/Disable tests
<tr><td>`ARCANE_ENABLE_DOTNET_WRAPPER` <td>`ON`/`OFF` <td>Enable/Disable the C#/.Net wrapper
<tr><td>`ARCANE_ENABLE_ALEPH` <td>`ON`/`OFF` <td>Enable/Disable support for the Aleph component
</table>
</details>

## Advanced compilation

### Accelerator support

<details>
<summary>Compilation options for accelerators</summary>
<table>
<tr><th>Option <th>Value <th>Description
<tr>
  <td>
    `ARCANE_ACCELERATOR_MODE`
  </td>
  <td>
    - `CUDANVCC` for NVIDIA GPUs
    - `ROCMHIP` for AMD GPUs
  </td>
  <td>
    Allows specifying the type of accelerator you wish to use.
    Starting from version 3.14 of %Arcane, it is possible to use `CUDA` instead
    of `CUDANVCC` and `ROCM` instead of `ROCMHIP`
  </td>
</tr>

<tr>
<td>`CMAKE_CUDA_COMPILER` <td>CUDA compiler (example: `nvcc` or `clang++`) </td>
<td>
Allows specifying the path to the historical CUDA compiler (`nvcc`) or another
compiler supporting the `ptx` format
</td>
</tr>

<tr>
<td>`CMAKE_HIP_COMPILER` <td>ROCM/HIP compiler (example: `amdclang++` or
`clang++`)</td>
<td>
Allows specifying the path to the compiler used to generate code for ROCM/HIP
</td>
</tr>

<tr>
<td>`CMAKE_CUDA_ARCHITECTURES` <td>Target architecture (example: `80`)</td>
<td>
Allows specifying a target architecture (Compute Capability). A list of multiple
values is possible (for example `80;90`)
</td>
</tr>

<tr>
<td>`CMAKE_HIP_ARCHITECTURES` <td>Target architecture (example: `gfx90a`)</td>
<td>Allows specifying a target architecture for AMD GPUs. A list of values is
possible (for example `gfx90a;gfx1031`)
</td>
</tr>

</table>
</details>

\note Before version 4.0 of %Arcane, it is necessary for accelerator support to
specify the CMake option `-DARCCORE_CXX_STANDARD=20`.

The CMake variable `ARCANE_ACCELERATOR_MODE` allows specifying the type of
accelerator you wish to use. There are currently two supported values:

- `CUDANVCC` or `CUDA` for NVIDIA GPUs
- `ROCMHIP` or `ROCM` for AMD GPUs

#### CUDA compilation

You must have at least version 12 of
[CUDA](https://developer.nvidia.com/cuda-downloads).

If you wish to compile CUDA support, you must add the argument
`-DARCANE_ACCELERATOR_MODE=CUDA` to the configuration and specify the path to
the `nvcc` or `clang++` compiler via the CMake variable `CMAKE_CUDA_COMPILER` or
the environment variable `CUDACXX`:

\warning If you wish to use %Arcane on both GPU and CPU, it is strongly
recommended to use `clang` as the compiler instead of `nvcc` because the latter
generates less performant code on the CPU side. This is due to the use of
`std::function` to encapsulate the lambdas used in %Arcane
(see [New Compiler Features in CUDA 8](https://developer.nvidia.com/blog/new-compiler-features-cuda-8/#extended___host_____device___lambdas)
for more information).

~~~{.sh}
# With 'clang'
cmake -DARCANE_ACCELERATOR_MODE=CUDA
-DCMAKE_CUDA_COMPILER=/usr/bin/clang++-19 \
...
~~~

~~~{.sh}
# With 'nvcc'
cmake -DARCANE_ACCELERATOR_MODE=CUDA
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc \
...
~~~

It is also possible to use the
NVIDIA [HPC SDK](https://developer.nvidia.com/hpc-sdk) compiler directly:

~~~{.sh}
export CXX=`which nvc++`
export CC=`which nvc`
cmake -DARCANE_ACCELERATOR_MODE=CUDA \
...
~~~

It is possible to specify a target architecture (Compute Capability) via the
`CMAKE_CUDA_ARCHITECTURES` variable, for example
`-DCMAKE_CUDA_ARCHITECTURES=80`.

#### AMD ROCM/HIP compilation

The minimum required ROCM version is 5.7.

To compile for AMD GPUs (such as the MI100 or MI250 GPUs), you must first
install the [ROCM](https://docs.amd.com/) library. When configuring %Arcane, you
must specify `-DARCANE_ACCELERATOR_MODE=ROCMHIP`.

For example, if ROCM is installed in `/opt/rocm` and you wish to compile for the
MI250 cards (gfx90x architecture):

~~~{.sh}
export ROCM_ROOT=/opt/rocm-5.0.0-9257
export CC=/opt/rocm/llvm/bin/clang
export CXX=/opt/rocm/llvm/bin/clang++
export CMAKE_HIP_COMPILER=/opt/rocm/hip/bin/hipcc

cmake -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/hip" \
-DARCANE_ACCELERATOR_MODE=ROCMHIP \
-DCMAKE_HIP_ARCHITECTURES=gfx90a \
...
~~~

### Documentation generation

<details>
<summary>Compilation options for documentation generation</summary>
<table>
<tr><th>Option <th>Value <th>Description
<tr><td>`ARCANEDOC_OFFLINE` <td>`ON` or `OFF` <td>Allows knowing if internet access is available
<tr><td>`ARCANEDOC_LEGACY_THEME` <td>`ON` or `OFF` <td>Allows generating the documentation with the original Doxygen style
</table>
</details>

Documentation generation has only been tested on Linux platforms.
It requires [Doxygen](https://www.doxygen.nl/index.html).

If `ARCANEDOC_OFFLINE=ON`, Doxygen requires a
[LaTeX](https://www.latex-project.org/) installation to correctly generate
certain equations.

Depending on the platform, it may be necessary to install additional LaTeX
packages (for example, the `texlive-latex-extra` package is required for
Ubuntu).

For configuration, two optional options are available:
- `ARCANEDOC_LEGACY_THEME`
- `ARCANEDOC_OFFLINE`

Each has two possible values: `ON` and `OFF`.

If the variables are not present, `OFF` is the default value.

Example:
```bash
cmake
  -S ... \
  -B ... \
  -DARCANEDOC_LEGACY_THEME=ON \
  -DARCANEDOC_OFFLINE=ON
```
The `ARCANEDOC_LEGACY_THEME` option allows generating the documentation with the
original Doxygen theme.

The `ARCANEDOC_OFFLINE` option tells CMake that the documentation will be used
locally, without internet access. This allows disabling elements that require
internet access, such as MathJax.

Once the configuration is complete, simply run:

For user documentation:

~~~{.sh}
cmake --build ${BUILD_DIR} --target userdoc
~~~

For developer documentation

~~~{.sh}
cmake --build ${BUILD_DIR} --target devdoc
~~~

The user documentation only contains information on classes useful for the
developer.

### Package search

<details>
<summary>Compilation options for package searching</summary>
<table>
<tr><th>Option <th>Value <th>Description
<tr><td>`ARCANE_NO_DEFAULT_PACKAGE` <td>`TRUE` or `FALSE` <td>Allows removing automatic package detection
<tr><td>`ARCANE_REQUIRED_PACKAGE_LIST` <td>Package name (example: `LibUnwind;HDF5`) <td>Allows explicitly specifying the
packages you wish to have
</table>
</details>

By default, all optional packages are automatically detected. It is possible to
remove this behavior and disable automatic package detection by adding
`-DARCANE_NO_DEFAULT_PACKAGE=TRUE` to the command line. In this case, you must
explicitly specify the packages you wish to have by listing them in the
`ARCANE_REQUIRED_PACKAGE_LIST` variable.
For example, if you only want `HDF5` and `LibUnwind` available, you must use
CMake as follows:

~~~{.sh}
cmake -DARCANE_NO_DEFAULT_PACKAGE=TRUE -DARCANE_REQUIRED_PACKAGE_LIST="LibUnwind;HDF5"
~~~

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_build_install_prerequisites
</span>
</div>
