# Projet ALIEN par IFPEN basé sur la version opensource

## Chargement des logiciels

``` 
#Attention en bash utilisez les fichier .sh

source ~commonlib/toolchain.csh
source ~commonlib/arcdev.csh
source ~commonlib/arcuser.csh
source ~commonlib/arcsolver-advanced.csh
module load CMake/3.14.3
module load dotNET-Core-Sdk/3.0.100
module load SLEPc/3.10.2

# à intégrer dans les modules
setenv SLEPc_ROOT ${SLEPC_DIR}
setenv SuperLU_ROOT ${SUPERLU_ROOT}
```

## Compilation des outils `arcane`

* `arccon` branche `dev-cea` dans le chemin `${Arccon_DIR}` :
```
git clone git@gitlab.ifpen.fr:Arcane/arccon.git
cd arccon
git checkout dev-cea
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=${Arccon_DIR} ..
make install
```

* `arccore` branche `dev-cea` dans le chemin `${Arccore_DIR}` :
```
git clone git@gitlab.ifpen.fr:Arcane/arccore.git
cd arccore
git checkout dev-cea
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release 
      -DCMAKE_INSTALL_PREFIX=${Arccore_DIR}
      -DArccon_DIR=${Arccon_DIR}/share/cmake/Arccon 
      -DBUILD_SHARED_LIBS=ON ..
make -j 8 install
```

* `arcdependencies` branche `dev-cea` dans le chemin `${ArcDependencies_DIR}` :
```
git clone git@gitlab.ifpen.fr:Arcane/dependencies.git
cd dependencies
git checkout dev-cea
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release 
      -DCMAKE_INSTALL_PREFIX=${ArcDependencies_DIR} 
      -DArccon_DIR=${Arccon_DIR}/share/cmake/Arccon ..
make install
```

* `axlstar` branche `dev-cea` dans le chemin `${Axlstar_DIR}` :
```
git clone git@gitlab.ifpen.fr:Arcane/axlstar.git
cd axlstar
git checkout dev-cea
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release       
      -DCMAKE_INSTALL_PREFIX=${Axlstar_DIR}       
      -DArccon_DIR=${Arccon_DIR}/share/cmake/Arccon       
      -DArcDependencies_DIR=${ArcDependencies_DIR}/share/cmake/ArcDependencies .. 
make install
```

## Compilation de `arcane`

* `arcane` branche `dev-cea` dans le chemin `${Arcane_DIR}` :
```
git clone git@gitlab.ifpen.fr:Arcane/arcane.git
cd arcane
git checkout dev-cea
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=${Arcane_DIR}
      -DArccon_DIR=${Arccon_DIR}/share/cmake/Arccon
      -DArccore_DIR=${Arccore_DIR}/lib/cmake/Arccore
      -DAxlstar_DIR=${Axlstar_DIR}/share/cmake/Axlstar
      -DArcDependencies_DIR=${ArcDependencies_DIR}/share/cmake/ArcDependencies 
      -DBUILD_SHARED_LIBS=ON 
      -DARCANE_DEFAULT_PARTITIONER=Metis
      -DARCANE_WANT_TOTALVIEW=ON
      -DARCANE_WANT_LIBXML2=ON
      -DARCANE_WANT_LEGACY_CONNECTIVITY=OFF
      -DARCANE_WANT_CHECK=OFF
      -DARCANE_WANT_ARCCON_EXPORT_TARGET=OFF ..
make -j 8 install
```


# Compilation `ALIEN` opensource

* `ALIEN` branche `open-source` dans le chemin `${Alien_DIR}` :
```
git clone git@gitlab.ifpen.fr:Arcane/alienopensource/alien.git
cd alien
git checkout open-source
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release 
      -DCMAKE_INSTALL_PREFIX=${Alien_DIR}
      -DBUILD_SHARED_LIBS=ON 
      -DArccon_DIR=${Arccon_DIR}/share/cmake/Arccon 
      -DArccore_DIR=${Arccore_DIR}/lib/cmake/Arccore       
      -DPLUGIN_DIR=${Plugin_DIR} ..
make -j 8 install
```

> Le répertoire `${Plugin_DIR}` de la macro `PLUGIN_DIR` permet de définir les plugins 
> opensource d'extensions de la plateforme `ALIEN`. Voir par exemple :
> * https://gitlab.ifpen.fr/Arcane/alienopensource/hypre
> * https://gitlab.ifpen.fr/Arcane/alienopensource/superlu  

# Compilation

```
cmake -DCMAKE_BUILD_TYPE=Release 
      -DCMAKE_INSTALL_PREFIX=${AlienProd_DIR}
      -DBUILD_SHARED_LIBS=ON 
      -DArccon_DIR=${Arccon_DIR}/share/cmake/Arccon 
      -DArccore_DIR=${Arccore_DIR}/lib/cmake/Arccore
      -DAxlstar_DIR=${Axlstar_DIR}/share/cmake/Axlstar
      -DAlien_DIR=${Alien_DIR}/lib/cmake/Alien      
      -DArcane_DIR=${Arcane_DIR}/lib/cmake/Arcane
      -DPLUGIN_DIR=${Plugin_DIR} ..
make -j 8 install
```
