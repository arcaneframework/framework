Exemple minimal d'utilisation {#arcanedoc_minimal_sample}
================== 

[TOC]

Ce chapître montre un exemple minimal d'utilisation de %Arcane.

Une application utilisant %Arcane est appelée un **Code**. Tout code doit
posséder au moins un module. Si on considère que le code s'appelle
`${code_name}` et le module `${module_name}`, alors une application minimale a besoin des
fichiers suivants:

- Le fichier de configuration du code permettant par exemple de
  spécifier les boucles en temps `${code_name}.config`
- Le fichier de description du module `${module_name}.axl`
- Le source C++ du module `${module_name}Module.cc`
- Le jeu de données `Test.arc`
- Le fichier principal du programme `main.cc`
- Le fichier `CMakeLists.txt` de configuration pour `cmake`

Le paragraphe \ref arcanedoc_generating_and_compiling_minimal montre
comment créer ces fichiers et compiler cette application minimale.

# Liste des fichiers générés

## Fichier de configuration du code

Ce fichier doit s'appeler du nom du code (comme spécifier dans le
main) et avoir l'extension `.config`.

Cet exemple minimal propose une boucle en temps de nom
`${code_name}Loop` qui utilise un seul module de nom `${module_name}`
et possède deux points d'entrée:

- un point d'entrée d'initialisation, nommé `StartInit`
- un point d'entrée appelé à chaque itération, nommé `Compute`

~~~{.xml}
<?xml version="1.0" ?>
<arcane-config code-name="${code_name}">
  <time-loops>
    <time-loop name="${code_name}Loop">
      <title>${module_name}</title>
      <description>Default timeloop for code ${code_name}</description>

      <modules>
        <module name="${module_name}" need="required" />
      </modules>

      <entry-points where="init">
        <entry-point name="${module_name}.StartInit" />
      </entry-points>
      <entry-points where="compute-loop">
        <entry-point name="${module_name}.Compute" />
      </entry-points>
    </time-loop>
  </time-loops>
</arcane-config>
~~~

## Descripteur AXL du module

Le fichier contenant le descripteur du module doit avoir pour nom
`${module_name}.axl`. Cet exemple montre un module avec deux points d'entrée.

~~~{.xml}
<?xml version="1.0" ?>
<module name="${module_name}" version="1.0">
  <description>Descripteur du module ${module_name}</description>
  <entry-points>
    <entry-point method-name="compute" name="Compute" where="compute-loop" property="none" />
    <entry-point method-name="startInit" name="StartInit" where="start-init" property="none" />
  </entry-points>
</module>
~~~

## Source C++ du module

Le fichier C++ contenant le code du module peut avoir n'importe quel
nom mais par convention on l'appelle `${module_name}Module.cc`. Ce
fichier doit contenir l'implémentation des deux points d'entrée
définis dans le descripteur de module. Ces points d'entrée
correspondante aux méthodes `startInit()` et `compute()`.

~~~{.cpp}
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-

#include "${module_name}_axl.h"
#include <arcane/ITimeLoopMng.h>

using namespace Arcane;

/*!
 * \brief Module ${module_name}.
 */
class ${module_name}Module
: public Arcane${module_name}Object
{
 public:
  explicit ${module_name}Module(const ModuleBuildInfo& mbi) 
  : Arcane${module_name}Object(mbi) { }

 public:
  /*!
   * \brief Méthode appelée à chaque itération.
   */
  void compute() override;
  /*!
   * \brief Méthode appelée lors de l'initialisation.
   */
  void startInit() override;

  /** Retourne le numéro de version du module */
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }
};

void ${module_name}Module::
compute()
{
  info() << "Module ${module_name} COMPUTE";

  // Stop code after 10 iterations
  if (m_global_iteration()>10)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

void ${module_name}Module::
startInit()
{
  info() << "Module ${module_name} INIT";
}

ARCANE_REGISTER_MODULE_${module_name_uppercase}(${module_name}Module);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
~~~

## Fichier principal du programme

Le fichier principal du programme décrit au minimum le nom du code et
sa version.
Fichier principal (main.cc)

~~~{.cpp}
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include <arcane/launcher/ArcaneLauncher.h>

using namespace Arcane;

int
main(int argc,char* argv[])
{
  ArcaneLauncher::init(CommandLineArguments(&argc,&argv));
  auto& app_build_info = ArcaneLauncher::applicationBuildInfo();
  app_build_info.setCodeName("${code_name}");
  app_build_info.setCodeVersion(VersionInfo(1,0,0));
  return ArcaneLauncher::run();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
~~~


## Jeu de données

Ce fichier est un exemple de jeu de données. Son extension est `.arc`.
La balise racine doit contenir deux attributes `codename` et
`codeversion` dont les valeurs doivent correspondrent à celle
spécifier dans le fichier `main.cc`.

Cet exemple utilise un maillage cartésien 2D contenant 20x20 mailles.

~~~{.xml}
<?xml version="1.0"?>
<case codename="${code_name}" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Sample</title>
    <timeloop>${code_name}Loop</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Cartesian2D" >
        <nb-part-x>1</nb-part-x> 
        <nb-part-y>1</nb-part-y>
        <origin>0.0 0.0</origin>
        <x><n>20</n><length>2.0</length></x>
        <y><n>20</n><length>2.0</length></y>
      </generator>
    </mesh>
  </meshes>

</case>
~~~


## Fichier de configuration pour CMake

Fichier `CMakeLists.txt`

~~~{.txt}
cmake_minimum_required(VERSION 3.16)
project(${code_name} LANGUAGES CXX)

find_package(Arcane REQUIRED)

add_executable(${code_name} ${module_name}Module.cc main.cc ${module_name}_axl.h)

arcane_generate_axl(${module_name})
arcane_add_arcane_libraries_to_target(${code_name})
target_include_directories(${code_name} PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
~~~

# Génération et compilation d'une application minimale {#arcanedoc_generating_and_compiling_minimal}

%Arcane fournit un script `arcane-templates` pour générer
automatiquement ces fichiers. Ce script se trouve dans le répertoire
`bin` de l'installation de %Arcane et s'utilise comme suit:

~~~{.sh}
arcane-templates generate-application --code-name MyCode --module-name MyModule --ouput-directory /tmp/test_code
~~~

Pour compiler le code, il suffit de se placer dans le répertoire
contenant ce fichier et de lancer la commande:

~~~{.sh}
cmake -DCMAKE_PREFIX_PATH=/path/to/arcane/install .
~~~

Il est ensuite de possible de compiler et d'exécuter le test:

~~~{.sh}
make
./${code_name} Test.arc
~~~
