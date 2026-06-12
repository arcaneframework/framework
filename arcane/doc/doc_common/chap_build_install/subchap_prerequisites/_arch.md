# ArchLinux {#arcanedoc_build_install_prerequisites_arch}

[TOC]

The following commands allow you to install CMake, .Net, and the dependencies
required for %Arcane (as well as the optional dependencies `TBB`, `HDF5`, and
`ParMetis`):

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
