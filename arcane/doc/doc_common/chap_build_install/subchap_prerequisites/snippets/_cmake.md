[//]: # (Il faut mettre le chemin du dossier dans EXAMPLE_PATH du doxyfile)

//![snippet_build_install_prerequisites_cmake]

La commande suivante permet d'installer la version 3.27.8 dans `/usr/local`.
Il faudra ensuite ajouter le chemin correspondant dans la variable d'environnement `PATH`.

~~~{sh}
# Install CMake 3.27.8 in /usr/local/cmake
cd /tmp
ARCH=`uname -m`
wget -O install.sh https://github.com/Kitware/CMake/releases/download/v3.27.8/cmake-3.27.8-linux-${ARCH}.sh
chmod u+x install.sh
./install.sh --skip-license --prefix=/usr/local
cmake --version
~~~

//![snippet_build_install_prerequisites_cmake]
