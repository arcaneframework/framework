# Prérequis {#arcanedoc_build_install_prerequisites}

Pour compiler et utiliser %Arcane, il est nécessaire d'installer certaines
dépendances. Ce sous-chapitre est dédié à l'installation de ces dépendances.

\note A partir de la version 4 de %Arcane, le support du C++20 est obligatoire.

<details>
<summary>Liste des dépendances nécessaires</summary>
<table>
<tr><th>Nom de la dépendance <th>Version (Mini/Maxi) <th>Description
<tr><td>[GCC](https://gcc.gnu.org/) <td>11/ <td rowspan="3">Compilateur supportant le C++20
<tr><td>[CLang](https://clang.llvm.org/) <td>15/
<tr><td>[Visual Studio](https://visualstudio.microsoft.com/) <td>17.4/
<tr><td>[Make](https://www.gnu.org/software/make/) <td> <td>Système de génération d'exécutable
<tr><td>[CMake](https://cmake.org/) <td>3.21/ (3.26/ si utilisation de CUDA) <td>Système de build de projet
<tr><td>[DotNet](https://dotnet.microsoft.com/) <td>8/ <td>Pour la partie C#
<tr><td>[GLib](https://www.gtk.org/) <td> <td>Support du multi-threading
<tr><td>[LibXml2](http://www.xmlsoft.org/) <td> <td>Lecture des fichiers AXL/ARC
</table>
</details>


<details>
<summary>Liste des dépendances recommandées</summary>
<table>
<tr><th>Nom de la dépendance <th>Version (Mini/Maxi) <th>Description
<tr><td>[oneTBB](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html) <td>2021/ <td>Support du multi-threading
<tr><td>[OpenMPI](https://www.open-mpi.org/) <td>4.0/ <td rowspan="2">Ou une autre implémentation qui supporte la norme MPI 3.1
<tr><td>[MPICH](https://www.mpich.org/) <td>3.2/
<tr><td>[ParMetis](https://github.com/KarypisLab/ParMETIS) </td><td>4.0</td><td>Partitionneur de graphe pour l'équilibrage de charge<td></tr>
</table>
</details>


<details>
<summary>Liste des dépendances optionnelles</summary>
<table>
<tr><th>Nom de la dépendance</th><th>Version (Mini/Maxi)</th><th>Description</th></tr>
<tr><td>[HDF5](https://www.hdfgroup.org/solutions/hdf5/) </td> <td>1.10/ </td><td>Bibliothèque de stockage de données</td></tr>
<tr><td>[Google Test](https://github.com/google/googletest) </td> <td>1.10</td> <td>Bibliothèque de tests unitaires</td></tr>
<tr><td>[Ninja](https://ninja-build.org/) </td><td>1.10/ </td><td>Système de génération d'exécutable</td></tr>
<tr><td>[SWIG](https://www.swig.org/) </td><td>4.1/ </td><td>Bibliothèque permettant d'appeler le C++ d'%Arcane avec du C#</td></tr>
<tr><td>[Hypre](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods) </td><td>2.20</td><td>Bibliothèque d'algèbre linéaire</td></tr>
<tr><td>[PETSc](https://gitlab.com/petsc/petsc) </td> <td>3.6</td><td>Bibliothèque d'algèbre linéaire</td></tr>
<tr><td>[Trilinos](https://trilinos.github.io/) </td> <td>16</td><td>Bibliothèque d'algèbre linéaire et de partitionnement de graphe</td></tr>
<tr><td>[Doxygen](https://doxygen.nl/) </td> <td>1.9.1/1.13.2 </td><td>Génération de la documentation</td></tr>

</table>
</details>

Choisissez votre OS :

- \subpage arcanedoc_build_install_prerequisites_ubuntu22 <br>
  Présente les prérequis nécessaires pour %Arcane sous Ubuntu 22.04.

- \subpage arcanedoc_build_install_prerequisites_ubuntu24 <br>
Présente les prérequis nécessaires pour %Arcane sous Ubuntu 24.04.

- \subpage arcanedoc_build_install_prerequisites_arch <br>
  Présente les prérequis nécessaires pour %Arcane sous ArchLinux.

- \subpage arcanedoc_build_install_prerequisites_rh9 <br>
  Présente les prérequis nécessaires pour %Arcane sous AlmaLinux 9 ou RedHat 9.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_build_install
</span>
<span class="next_section_button">
\ref arcanedoc_build_install_build
</span>
</div>
