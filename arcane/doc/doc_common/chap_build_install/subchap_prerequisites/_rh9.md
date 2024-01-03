# AlmaLinux/RedHat 9 {#arcanedoc_build_install_prerequisites_rh9}

[TOC]

## Installation des packages nécessaires

Pour compiler Arcane, il est nécessaire d'installer les packages
suivants:

~~~{sh}
# Packages nécessaires
yum -y install dotnet-sdk-6.0 glib2-devel libxml2-devel gcc-c++
~~~

La version par défaut de CMake sur AlmaLinux 9 est trop ancienne
(3.20). Il faut donc télécharger une version plus récente.

\snippet{doc} _cmake.md snippet_build_install_prerequisites_cmake


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_build_install_prerequisites
</span>
<span class="next_section_button">
\ref arcanedoc_build_install_build
</span>
</div>
