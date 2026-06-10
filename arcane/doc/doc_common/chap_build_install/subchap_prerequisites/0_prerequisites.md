# Prerequisites {#arcanedoc_build_install_prerequisites}

To compile and use %Arcane, it is necessary to install certain dependencies.
This subsection is dedicated to installing these dependencies.

\note Starting from version 4 of %Arcane, C++20 support is mandatory.

<details>
<summary>List of required dependencies</summary>
<table>
<tr><th>Dependency Name <th>Version (Min/Max) <th>Description
<tr><td>[GCC](https://gcc.gnu.org/)</td> <td>11/</td> <td rowspan="3">Compiler supporting C++20</td></tr>
<tr><td>[CLang](https://clang.llvm.org/)</td> <td>15/</td></tr>
<tr><td>[Visual Studio](https://visualstudio.microsoft.com/)</td> <td>17.4/</td></tr>
<tr><td>[Make](https://www.gnu.org/software/make/)</td> <td> <td>Executable generation system</td></tr>
<tr><td>[CMake](https://cmake.org/)</td> <td>3.21/ (3.26/ if using CUDA) <td>Project build system</td></tr>
<tr><td>[DotNet](https://dotnet.microsoft.com/)</td> <td>8/</td> <td>For AxlStar and the C# wrapper</td></tr>
<tr><td>[GLib](https://www.gtk.org/)</td> <td> </td> <td>Multi-threading support</td></tr>
<tr><td>[LibXml2](http://www.xmlsoft.org/)</td> <td> </td> <td>Reading AXL/ARC files</td></tr>
</table>
</details>

<details>
<summary>List of recommended dependencies</summary>
<table>
<tr><th>Dependency Name <th>Version (Min/Max) <th>Description
<tr><td>[oneTBB](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html) <td>2021/ <td>Multi-threading support
<tr><td>[OpenMPI](https://www.open-mpi.org/) <td>4.0/ <td rowspan="2">Or another implementation that supports the MPI 3.1 standard
<tr><td>[MPICH](https://www.mpich.org/) <td>3.2/
<tr><td>[ParMetis](https://github.com/KarypisLab/ParMETIS) </td><td>4.0</td><td>Graph partitioner for load balancing<td></tr>
</table>
</details>

<details>
<summary>List of optional dependencies</summary>
<table>
<tr><th>Dependency Name</th><th>Version (Min/Max)</th><th>Description</th></tr>
<tr><td>[HDF5](https://www.hdfgroup.org/solutions/hdf5/) </td> <td>1.10/ </td><td>Data storage library</td></tr>
<tr><td>[Google Test](https://github.com/google/googletest) </td> <td>1.10</td> <td>Unit testing library</td></tr>
<tr><td>[Ninja](https://ninja-build.org/) </td><td>1.10/ </td><td>Executable generation system</td></tr>
<tr><td>[SWIG](https://www.swig.org/) </td><td>4.1/ </td><td>Library allowing C++ in %Arcane to be called from C#</td></tr>
<tr><td>[Hypre](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods) </td><td>2.20</td><td>Linear algebra library</td></tr>
<tr><td>[PETSc](https://gitlab.com/petsc/petsc) </td> <td>3.6</td><td>Linear algebra library</td></tr>
<tr><td>[Trilinos](https://trilinos.github.io/) </td> <td>16</td><td>Linear algebra and graph partitioning library</td></tr>
<tr><td>[Doxygen](https://doxygen.nl/) </td> <td>1.9.1/1.13.2 </td><td>Documentation generation</td></tr>

</table>
</details>

Choose your OS:

- \subpage arcanedoc_build_install_prerequisites_ubuntu22 <br>
  Presents the prerequisites necessary for %Arcane under Ubuntu 22.04.

- \subpage arcanedoc_build_install_prerequisites_ubuntu24 <br>
  Presents the prerequisites necessary for %Arcane under Ubuntu 24.04.

- \subpage arcanedoc_build_install_prerequisites_ubuntu26 <br>
  Presents the prerequisites necessary for %Arcane under Ubuntu 26.04.

- \subpage arcanedoc_build_install_prerequisites_arch <br>
  Presents the prerequisites necessary for %Arcane under ArchLinux.

- \subpage arcanedoc_build_install_prerequisites_rh9 <br>
  Presents the prerequisites necessary for %Arcane under AlmaLinux 9 or RedHat
  9.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_build_install
</span>
<span class="next_section_button">
\ref arcanedoc_build_install_build
</span>
</div>
