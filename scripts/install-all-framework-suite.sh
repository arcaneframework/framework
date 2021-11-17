user1=$1
password1=$2
user2=$3
password2=$4
INSTALL_DIR=/usr/local/appli
Arccon_DIR=${INSTALL_DIR}/arccon
Arccore_DIR=${INSTALL_DIR}/arccore
Axlstar_DIR=${INSTALL_DIR}/axlstar
Arcane_DIR=${INSTALL_DIR}/arcane
Alien_DIR=${INSTALL_DIR}/alien 
AlienProd_DIR=${INSTALL_DIR}/alien-prod
ArcDependencies_DIR=${INSTALL_DIR}/dependencies
CLONE_FIRST=ON
COMPIL_PACKAGE=ON
BUILD_ARCCON=ON
BUILD_ARCCORE=ON
BUILD_ARCDEP=ON
BUILD_AXLSTAR=ON
BUILD_ARCANE=ON
BUILD_ALIEN=ON
BUILD_ALIENLEGACY=ON
BUILD_ARCANEDEMO=ON
if [ -n "${BUILD_ARCCON}" ] ; then
echo "INSTALL ARCCON"
if [ -n "${CLONE_FIRST}" ] ; then
echo "      CLONE PROJECT"
git clone https://${user1}:${password1}@gitlab.com/cea-ifpen/arccon.git
fi
if [ -n "${COMPIL_PACKAGE}" ] ; then
cd arccon
git checkout main
\rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${Arccon_DIR} \
      ..
make install
cd ../..
fi
else
echo "SKIP INSTALL ARCCON"
fi

if [ -n "${BUILD_ARCCORE}" ] ; then
echo "INSTALL ARCCORE"
if [ -n "${CLONE_FIRST}" ] ; then
echo "      CLONE PROJECT"
git clone https://${user1}:${password1}@gitlab.com/cea-ifpen/arccore.git
fi
if [ -n "${COMPIL_PACKAGE}" ] ; then
cd arccore
echo "COMPIL ARCCORE"
git checkout main
\rm -rf build
mkdir build
cd build
echo "CONFIGURE ARCCORE"
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=${Arccore_DIR} \
      -DArccon_DIR=${Arccon_DIR}/share/cmake/Arccon \
      -DBUILD_SHARED_LIBS=ON ..
echo "COMPILING ARCCORE"
make -j 8 install
#make test
cd ../..
fi
else
echo "SKIP INSTALL ARCCORE"
fi


if [ -n "${BUILD_ARCDEP}" ] ; then
echo "INSTALL ARCANEDEPENDENCIES"
if [ -n "${CLONE_FIRST}" ] ; then
echo "      CLONE PROJECT"
git clone https://${user2}:${password2}@gitlab.ifpen.fr/Arcane/dependencies.git
fi
if [ -n "${COMPIL_PACKAGE}" ] ; then
cd dependencies
\rm -rf build
mkdir build
cd build
echo "CONFIGURE PACKAGE"
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=${ArcDependencies_DIR} \
      -DArccon_DIR=${Arccon_DIR}/share/cmake/Arccon ..
make install
cd ../..
fi
else
echo "SKIP INSTALL ARCDEP"
fi

if [ -n "${BUILD_AXLSTAR}" ] ; then
echo "INSTALL AXLSTAR"
if [ -n "${CLONE_FIRST}" ] ; then
echo "      CLONE PROJECT"
git clone https://${user2}:${password2}@gitlab.ifpen.fr/Arcane/axlstar.git
fi
if [ -n "${COMPIL_PACKAGE}" ] ; then
cd axlstar
\rm -rf build
mkdir build
cd build
echo "CONFIGURE PACKAGE"
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=${Axlstar_DIR} \
      -DArccon_DIR=${Arccon_DIR}/share/cmake/Arccon \
      -DArcDependencies_DIR=${ArcDependencies_DIR}/share/cmake/ArcDependencies ..
make install
cd ../..
fi
else
echo "SKIP INSTALL ARCAXLSTAR"
fi

if [ -n "${BUILD_ARCANE}" ] ; then
echo "INSTALL ARCANE"
if [ -n "${CLONE_FIRST}" ] ; then
echo "      CLONE PROJECT"
git clone https://${user2}:${password2}@gitlab.ifpen.fr/Arcane/arcane.git
fi
if [ -n "${COMPIL_PACKAGE}" ] ; then
cd arcane
\rm -rf build
mkdir build 
cd build
echo "CONFIGURE PACKAGE"
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=${Arcane_DIR} \
      -DArccon_DIR=${Arccon_DIR}/share/cmake/Arccon \
      -DArccore_DIR=${Arccore_DIR}/lib/cmake/Arccore \
      -DAxlstar_DIR=${Axlstar_DIR}/share/cmake/Axlstar \
      -DArcDependencies_DIR=${ArcDependencies_DIR}/share/cmake/ArcDependencies \
      -DBUILD_SHARED_LIBS=ON \
      -DARCANE_DEFAULT_PARTITIONER=Metis \
      -DARCANE_WANT_TOTALVIEW=ON \
      -DARCANE_WANT_LIBXML2=ON \
      -DARCANE_WANT_LEGACY_CONNECTIVITY=OFF \
      -DARCANE_WANT_CHECK=OFF \
      -DARCANE_WANT_ARCCON_EXPORT_TARGET=OFF ..
make -j 8 install
cd ../..
fi
else
echo "SKIP INSTALL ARCANE"
fi

if [ -n "${BUILD_ALIEN}" ] ; then
echo "INSTALL ALIEN"
if [ -n "${CLONE_FIRST}" ] ; then
echo "      CLONE PROJECT"
git clone https://${user1}:${password1}@gitlab.com/cea-ifpen/alien.git 
fi
if [ -n "${COMPIL_PACKAGE}" ] ; then
cd alien
git checkout main
\rm -rf build
mkdir build
cd build
echo "CONFIGURE PACKAGE"
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=${Alien_DIR} \
      -DBUILD_SHARED_LIBS=ON \
      -DArccon_DIR=${Arccon_DIR}/share/cmake/Arccon \
      -DArccore_DIR=${Arccore_DIR}/lib/cmake/Arccore \
      -DALIEN_GENERATE_DOCUMENTATION=OFF \
      -DALIEN_USE_LIBXML2=On \
      -DALIEN_USE_HDF5=On \
       ..
make -j 1 install
#make test
cd ../..
fi
else
echo "SKIP INSTALL ALIEN"
fi

if [ -n "${BUILD_ALIENLEGACY}" ] ; then
echo "INSTALL ALIEN-LEGACY"
if [ -n "${CLONE_FIRST}" ] ; then
echo "      CLONE PROJECT"
git clone https://${user1}:${password1}@gitlab.ifpen.fr/Arcane/ifpen-tools/alien-legacy.git
fi
if [ -n "${COMPIL_PACKAGE}" ] ; then
cd alien-legacy
git checkout master
git submodule update --init --recursive
#git submodule update 
\rm -rf build
mkdir build
cd build

echo "CONFIGURE PACKAGE"
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=${AlienProd_DIR} \
      -DBUILD_SHARED_LIBS=ON \
      -DArccon_DIR=${Arccon_DIR}/share/cmake/Arccon \
      -DArccore_DIR=${Arccore_DIR}/lib/cmake/Arccore \
      -DAxlstar_DIR=${Axlstar_DIR}/share/cmake/Axlstar \
      -DArcane_DIR=${Arcane_DIR}/lib/cmake/Arcane \
      -DAlien_DIR=${Alien_DIR}/lib/cmake/Alien \
      -DALIEN_GENERATE_DOCUMENTATION=On \
      -DUSE_BUILDSYSTEM_GIT_SUBMODULE=On \
      ..
make -j 1 install
#make test
cd ../..
fi
else
echo "SKIP INSTALL ALIENLEGACY"
fi

if [ -n "${BUILD_ARCANEDEMO}" ] ; then
echo "INSTALL ARCANEDEMO"
if [ -n "${CLONE_FIRST}" ] ; then
echo "      CLONE PROJECT"
svn co https://${user2}:${password2}@websvn.ifpen.fr/svn/ArcSim/ArcaneDemo
fi
if [ -n "${COMPIL_PACKAGE}" ] ; then
cd ArcaneDemo
\rm -rf build
mkdir build
cd build

echo "CONFIGURE PACKAGE"
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DBUILD_SHARED_LIBS=ON \
      -DArccon_DIR=${Arccon_DIR}/share/cmake/Arccon \
      -DArccore_DIR=${Arccore_DIR}/lib/cmake/Arccore \
      -DAxlstar_DIR=${Axlstar_DIR}/share/cmake/Axlstar \
      -DArcanePath=${Arcane_DIR} \
      -DAlienProd_DIR=${AlienProd_DIR}/lib/cmake \
      -DUSE_ALIEN_V20=ON \
      -DUSE_ARCANE_V3=0N \
      -DUSE_ARCCON=0N \
      -DUSE_BUILDSYSTEM_PROJECT=0N \
      -DArcane_DIR=${Arcane_DIR}/lib/cmake/Arcane \
      -DAlienPath=${AlienProd_DIR} \
      -DArcanePath=${Arcane_DIR} \
      ..
make -j 1 install
#make test
fi
cd ../..
else
echo "SKIP INSTALL ARCANEDEMO"
fi
