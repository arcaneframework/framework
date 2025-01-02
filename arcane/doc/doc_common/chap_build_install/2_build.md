# Compilation {#arcanedoc_build_install_build}

[TOC]

La compilation doit se faire dans un répertoire différent de celui
contenant les sources.

## Récupération des sources

Pour récupérer les sources :

~~~{sh}
git clone --recurse-submodules https://github.com/arcaneframework/framework
~~~

ou

~~~{sh}
git clone https://github.com/arcaneframework/framework
cd framework && git submodule update --init --recursive
~~~

## Compilation

Par défaut, on compile %Arcane et Alien si les pré-requis sont disponibles.
La variable CMake `ARCANEFRAMEWORK_BUILD_COMPONENTS` contient la liste
des composants du dépôt à compiler. Cette liste peut contenir les
valeurs suivantes:

- `%Arcane`
- `Alien`

Par défaut la valeur est `%Arcane;Alien` et donc on compile les deux composants.

Pour compiler %Arcane et Alien , il faut procéder comme suit:

~~~{sh}
mkdir /path/to/build
cmake -S /path/to/sources -B /path/to/build
cmake --build /path/to/build
cmake --build /path/to/build --target install
~~~

Par défaut, l'installation se fait dans `/usr/local` si l'option
`CMAKE_INSTALL_PREFIX` n'est pas spécifiée.

<details>
<summary>Les options de compilation générales disponibles</summary>
Options générales :
<table>
<tr><th>Option <th>Valeur <th>Description
<tr><td>`CMAKE_INSTALL_PREFIX` <td>`/path/to/install` <td>Choix d'un dossier d'installation
<tr><td>`ARCANEFRAMEWORK_BUILD_COMPONENTS` <td>`%Arcane` ou `Alien` ou `%Arcane;Alien` <td>Composant(s) à compiler 
<tr><td>`ARCCORE_CXX_STANDARD` <td>`17` ou `20` ou `23` <td>Choix du standard C++ à utiliser
</table>
</details>

## Compilation avancée

### Support des accélérateurs

<details>
<summary>Les options de compilation pour les accélérateurs</summary>
<table>
<tr><th>Option <th>Valeur <th>Description
<tr>
  <td>
    `ARCANE_ACCELERATOR_MODE`
  </td>
  <td>
    - `CUDANVCC` pour les GPU NVIDIA
    - `ROCMHIP` pour les GPU AMD
  </td>
  <td>
    Permet de spécifier le type d'accélerateur qu'on souhaite utiliser.
    A partir de la version 3.14 de %Arcane, il est possible d'utiliser `CUDA` au lieu
    de `CUDANVCC` et `ROCM` au lieu de `ROCMHIP`
  </td>
</tr>

<tr>
<td>`CMAKE_CUDA_COMPILER` <td>Compilateur CUDA (exemple : `nvcc` ou
`clang++`) </td>
<td>
Permet de spécifier le chemin vers le compilateur CUDA historique (`nvcc`) ou un autre
compilateur supportant le format `ptx`
</td>
</tr>

<tr>
<td>`CMAKE_HIP_COMPILER` <td>Compilateur ROCM/HIP (exemple :
`amdclang++` ou `clang++`)</td>
<td>
Permet de spécifier le chemin vers le compilateur utilisé pour générer
le code pour ROCM/HIP
</td>
</tr>

<tr>
<td>`CMAKE_CUDA_ARCHITECTURES` <td>Architecture cible (exemple : `80`)</td>
<td>
Permet de spécifier une architecture cible (Capability Compute). Une
liste de plusieurs valeurs est possible (par exemple `80;90`)
</td>
</tr>

<tr>
<td>`CMAKE_HIP_ARCHITECTURES` <td>Architecture cible (exemple : `gfx90a`)</td>
<td>Permet de spécifier une architecture cible pour les GPUS AMD.
Une liste de valeurs est possible (par exemple `gfx90a;gfx1031`)
</td>
</tr>

</table>
</details>

Depuis la version 3.12 de %Arcane, le support des accélérateurs nécessite un
compilateur supportant le C++20. Il est donc nécessaire de compiler
%Arcane en spécifiant la variable CMake `-DARCCORE_CXX_STANDARD=20`.

La variable CMake `ARCANE_ACCELERATOR_MODE` permet de spécifier le
type d'accélerateur qu'on souhaite utiliser. Il y a actuellement deux
valeurs supportées:

- `CUDANVCC` ou `CUDA` pour les GPU NVIDIA
- `ROCMHIP` ou `ROCM` pour les GPU AMD

#### Compilation CUDA

Il est nécessaire d'avoir au moins la version 12 de
[CUDA](https://developer.nvidia.com/cuda-downloads).

Si on souhaite compiler le support CUDA, il faut ajouter l'argument
`-DARCANE_ACCELERATOR_MODE=CUDA` à la configuration et spécifier
le chemin vers le compilateur `nvcc` ou `clang++` via la variable CMake
`CMAKE_CUDA_COMPILER` ou la variable d'environnement `CUDACXX`:

\warning Si on souhaite utiliser %Arcane à la fois sur GPU et sur CPU,
il est fortement recommandé d'utiliser `clang` comme compilateur au
lieu de `nvcc` car ce dernier génère du code moins performant sur la
partie CPU. Cela est du à l'usage de `std::function` pour encapsuler
les lambdas utilisées dans %Arcane (voir
[New Compiler Features in CUDA 8](https://developer.nvidia.com/blog/new-compiler-features-cuda-8/#extended___host_____device___lambdas)
pour plus d'informations)

~~~{.sh}
# Avec 'clang'
cmake -DARCANE_ACCELERATOR_MODE=CUDA
-DCMAKE_CUDA_COMPILER=/usr/bin/clang++-19 \
-DARCCORE_CXX_STANDARD=20 \
...
~~~

~~~{.sh}
# Avec 'nvcc'
cmake -DARCANE_ACCELERATOR_MODE=CUDA
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc \
-DARCCORE_CXX_STANDARD=20 \
...
~~~

Il est aussi possible d'utiliser directement le compilateur du [HPC
SDK](https://developer.nvidia.com/hpc-sdk) de NVidia:

~~~{.sh}
export CXX=`which nvc++`
export CC=`which nvc`
cmake -DARCANE_ACCELERATOR_MODE=CUDA \
-DARCCORE_CXX_STANDARD=20 \
...
~~~

Il est possible de spécifier une architecture cible (Capability
Compute) via la variable `CMAKE_CUDA_ARCHITECTURES`, par exemple
`-DCMAKE_CUDA_ARCHITECTURES=80`.

#### Compilation AMD ROCM/HIP

Pour compiler pour les GPU AMD (comme par exemple les GPU MI100 ou
MI250) il faut avoir auparavant installer la bibliothèque [ROCM](https://docs.amd.com/). Lors
de la configuration de %Arcane, il faut spécifier `-DARCANE_ACCELERATOR_MODE=ROCMHIP`.

Par exemple, si ROCM est installé dans `/opt/rocm` et qu'on souhaite
compiler pour les cartes MI250 (architecture gfx90x):

~~~{.sh}
export ROCM_ROOT=/opt/rocm-5.0.0-9257
export CC=/opt/rocm/llvm/bin/clang
export CXX=/opt/rocm/llvm/bin/clang++
export CMAKE_HIP_COMPILER=/opt/rocm/hip/bin/hipcc

cmake -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/hip" \
-DARCANE_ACCELERATOR_MODE=ROCMHIP \
-DCMAKE_HIP_ARCHITECTURES=gfx90a \
-DARCCORE_CXX_STANDARD=20 \
...
~~~



### Génération de la documentation

<details>
<summary>Les options de compilation pour la génération de la documentation</summary>
<table>
<tr><th>Option <th>Valeur <th>Description
<tr><td>`ARCANEDOC_OFFLINE` <td>`ON` ou `OFF` <td>Permet de savoir si on a accès à internet
<tr><td>`ARCANEDOC_LEGACY_THEME` <td>`ON` ou `OFF` <td>Permet de générer la documentation avec le style de Doxygen original
</table>
</details>

La génération de la documentation n'a été testée que sur les plateforme Linux.
Elle nécessite l'outil [Doxygen](https://www.doxygen.nl/index.html).

Si `ARCANEDOC_OFFLINE=ON`, l'outil Doxygen a besoin d'une installation de
[LaTeX](https://www.latex-project.org/) pour générer correctement
certaines équations.

Suivant les plateformes, il peut être nécessaire
d'installer des packages LaTeX supplémentaires (par exemple pour
Ubuntu, le package `texlive-latex-extra` est nécessaire).

Pour la configuration, deux options facultatives sont disponibles :
- `ARCANEDOC_LEGACY_THEME`
- `ARCANEDOC_OFFLINE`

Avec chaqu'une deux valeurs possibles : `ON` et `OFF`.

Si les variables ne sont pas présentes, `OFF` est la valeur par défaut.

Exemple :
```bash
cmake
  -S ... \
  -B ... \
  -DARCANEDOC_LEGACY_THEME=ON \
  -DARCANEDOC_OFFLINE=ON
```
L'option `ARCANEDOC_LEGACY_THEME` permet de générer la documentation
avec le thème d'origine de Doxygen.

L'option `ARCANEDOC_OFFLINE` permet de dire à CMake que la documentation
sera utilisée en local, sans accès à internet. Cela permet de désactiver
les élements ayant besoin d'un accès à internet, comme MathJax.

Une fois la configuration terminée, il suffit de lancer:

Pour la documentation utilisateur:

~~~{.sh}
cmake --build ${BUILD_DIR} --target userdoc
~~~

Pour la documentation développeur

~~~{.sh}
cmake --build ${BUILD_DIR} --target devdoc
~~~

La documentation utilisateur ne contient les infos que des classes
utiles pour le développeur.





### Recherche de packages

<details>
<summary>Les options de compilation pour la recherche de packages</summary>
<table>
<tr><th>Option <th>Valeur <th>Description
<tr><td>`ARCANE_NO_DEFAULT_PACKAGE` <td>`TRUE` ou `FALSE` <td>Permet de supprimer la détection automatique des packages
<tr><td>`ARCANE_REQUIRED_PACKAGE_LIST` <td>Nom de packages (exemple : `LibUnwind;HDF5`) <td>Permet de préciser explicitement les packages qu'on souhaite avoir
</table>
</details>

Par défaut, tous les packages optionnels sont détectés
automatiquement. Il est possible de supprimer ce comportement et de
supprimer la détection automatique des packages en ajoutant
`-DARCANE_NO_DEFAULT_PACKAGE=TRUE` à la ligne de commande. Dans ce
cas, il faut préciser explicitement les packages qu'on souhaite avoir
en les spécifiant à la variable `ARCANE_REQUIRED_PACKAGE_LIST` sous
forme de liste. Par exemple, si on souhaite avoir uniquement `HDF5` et
`LibUnwind` de disponible, il faut utilise CMake comme suit:

~~~{.sh}
cmake -DARCANE_NO_DEFAULT_PACKAGE=TRUE -DARCANE_REQUIRED_PACKAGE_LIST="LibUnwind;HDF5"
~~~

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_build_install_prerequisites
</span>
</div>
