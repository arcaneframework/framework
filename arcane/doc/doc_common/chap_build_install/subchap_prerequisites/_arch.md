# ArchLinux {#arcanedoc_build_install_prerequisites_arch}

[TOC]

Les commandes suivantes permettent d'installer CMake, .Net et les dépendances
nécessaires pour %Arcane (ainsi que les dépendances optionnelles `TBB`, `HDF5` et `ParMetis`):

~~~{sh}
sudo pacman -Syu
sudo pacman -S gcc cmake python git gcc-fortran glib2 libxml2 hdf5-openmpi wget tbb dotnet-sdk aspnet-runtime aspnet-targeting-pack
yay -S aur/parmetis
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
