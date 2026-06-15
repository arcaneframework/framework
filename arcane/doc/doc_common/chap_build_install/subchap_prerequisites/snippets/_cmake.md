[//]: # (You must put the folder path in the doxyfile's EXAMPLE_PATH)

//![snippet_build_install_prerequisites_cmake]

The following command allows installing version 3.27.8 in /usr/local.
You will then need to add the corresponding path to the `PATH` environment
variable.

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
