# AlmaLinux/RedHat 9 {#arcanedoc_build_install_prerequisites_rh9}

[TOC]

## Installation of necessary packages

To compile Arcane, it is necessary to install the following packages:

~~~{sh}
# Necessary packages
yum -y install dotnet-sdk-6.0 glib2-devel libxml2-devel gcc-c++
~~~

The default version of CMake on AlmaLinux 9 is too old (3.20). Therefore, you
must download a more recent version.

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
