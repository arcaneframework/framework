# Module SayHello {#arcanedoc_examples_simple_example_module}

[TOC]

Notre module `SayHello` est donc composé des trois fichiers ci-dessous :
- un header (.h),
- un fichier source (.cc),
- un fichier contenant les options du jeu de données (.axl).

\note Par convention, les noms des fichiers des modules suivent un schéma
précis :
- le nom du module est libre,
- les `headers` portent l'extension `.h` (et non `.hh`/`.hpp`/`.hxx`/&c) et sont composés
comme cela : `{nom_module}` `Module` `.h` (exemple : `SayHelloModule.h`),
- les `fichiers sources` portent l'extension `.cc` (et non `.c`/`.cxx`/`.c++`/&c) et sont composés
comme cela : `{nom_module}` `Module` `.cc` (exemple : `SayHelloModule.cc`),
- les `fichiers de configuration de jeu de données` (fichiers `AXL`) portent l'extension `.axl` 
et sont composés comme cela : `{nom_module}` `.axl` (exemple : `SayHello.axl`).

## SayHello.axl {#arcanedoc_examples_simple_example_module_sayhelloaxl}

D'abord, voyons le fichier `.axl` :

```xml
<?xml version="1.0" ?>
<module name="SayHello" version="1.0">

  <description>Descripteur du module SayHello</description>

  <variables>
    <variable 
      field-name="loop_sum"
      name="LoopSum"
      data-type="integer"
      item-kind="none"
      dim="0" />
  </variables>

  <entry-points>
    <entry-point method-name="startInit" name="StartInit" where="start-init" property="none" />
    <entry-point method-name="compute" name="Compute" where="compute-loop" property="none" />
    <entry-point method-name="endModule" name="EndModule" where="exit" property="none" />
  </entry-points>

  <options>
    <simple name="nSteps" type="integer" default="10">
      <description>Nombre de boucle à effectuer.</description>
    </simple>
  </options>

</module>
```
C'est un fichier au format `xml` qui permet de décrire le fonctionnement de
notre module `SayHello`.

On peut voir que ce fichier contient le nom du module (ligne 2), une description sommaire (ligne 4), des variables (lignes 6-13), des points d'entrées (lignes 15-19) et des options (lignes 21-25).

Les variables :
- La variable de nom "Arcane" `LoopSum` et de nom "code" `loop_sum`, de type `Integer`, attribuée à l'item `None` et de dimension `0`.

Les points d'entrées :
- Le point d'entrée de nom `StartInit`, représenté par la méthode `startInit` et s'exécutant à l'initialisation (`start-init`),
- Le point d'entrée de nom `Compute`, représenté par la méthode `compute` et s'exécutant dans la boucle en temps (`compute-loop`).
- Le point d'entrée de nom `EndModule`, représenté par la méthode `endModule` et s'exécutant lorsque la boucle en temps est terminée (`exit`).

Les options :
- L'option simple de nom `nSteps`, de type `Integer` et avec une valeur par défaut égale à `10`.

Le compilateur, à partir de ce fichier, créera automatiquement un fichier `SayHello_axl.h` qu'il faudra importer dans notre header et qui nous permettra d'utiliser :
- L'attribut `m_loop_sum` (`m_` + le `field-name`),
- La méthode `startInit`,
- La méthode `compute`,
- L'option `nStep`.


## SayHelloModule.h {#arcanedoc_examples_simple_example_module_sayhellomoduleh}

Voici à quoi ressemble notre header :

```cpp
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef SAYHELLOMODULE_H
#define SAYHELLOMODULE_H
 
#include <arcane/ITimeLoopMng.h>
#include "SayHello_axl.h"
 
using namespace Arcane;
 
class SayHelloModule
: public ArcaneSayHelloObject
{
 public:
  explicit SayHelloModule(const ModuleBuildInfo& mbi) 
  : ArcaneSayHelloObject(mbi) { }

 public:
  void startInit() override;
  void compute() override;
  void endModule() override;
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }
};
 
#endif
```

Entrons dans les détails.
```cpp
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
```
Dans le chapitre \ref arcanedoc_general_codingrules, on nous demande de mettre cette ligne nous informant de la configuration du fichier.

____

```cpp
#ifndef SAYHELLOMODULE_H
#define SAYHELLOMODULE_H
//...
#endif
```
Cela permet d'éviter de définir plusieurs fois la classe (si on l'inclut dans plusieurs `.cc` par exemple).

____

```cpp
#include <arcane/ITimeLoopMng.h>
#include "SayHello_axl.h"
```
Ce premier `#include` permet d'include les fonctions pour gérer la boucle en temps (par exemple `stopComputeLoop()`).
Le second `#include` est le fichier généré avec le `.axl`.

____

```cpp
using namespace Arcane;
```
Cela permet d'utiliser les fonctions d'%Arcane sans mettre `Arcane::` devant.

____

```cpp
class SayHelloModule
: public ArcaneSayHelloObject
```
On définie la classe `SayHelloModule` qui sera utilisée par %Arcane et qui hérite de `ArcaneSayHelloObject`. `ArcaneSayHelloObject` est définie dans `SayHello_axl.h` et contient les méthodes que l'on pourra override et les variables/options que l'on pourra utiliser.

____

```cpp
 public:
  explicit SayHelloModule(const ModuleBuildInfo& mbi) 
  : ArcaneSayHelloObject(mbi) { }
```
Constructeur de notre classe, qui appelle le constructeur de `ArcaneSayHelloObject`. `mbi` est un objet qui contiendra les infos de lancement du module (pour avoir les valeurs des options par exemple).

____

```cpp
 public:
  void startInit() override;
  void compute() override;
  void endModule() override;
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }
```
Enfin, les quatre méthodes du `.axl` que l'on override. On peut aussi donner une version à notre module en "overridant" `versionInfo()`.

____

## SayHelloModule.cc {#arcanedoc_examples_simple_example_module_sayhellomodulecc}

Ce fichier contient simplement les implémentations des méthodes définies dans le header.

```cpp
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-

#include "SayHelloModule.h"

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SayHelloModule::
startInit()
{
  info() << "Module SayHello INIT";
  m_loop_sum = 0;
}

void SayHelloModule::
compute()
{
  info() << "Module SayHello COMPUTE";

  m_loop_sum = m_loop_sum() + m_global_iteration();

  if (m_global_iteration() > options()->getNStep())
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

void SayHelloModule::
endModule()
{
  info() << "Module SayHello END";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_SAYHELLO(SayHelloModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
```

Plusieurs choses sont néanmoins remarquable ici :
```cpp
info() << "Module SayHello INIT";
```
`info()` permet d'écrire dans la sortie standard des informations.

____

```cpp
m_loop_sum = 0;
```
On attribue la valeur 0 à la variable `m_loop_sum`.

____

```cpp
m_loop_sum = m_loop_sum() + m_global_iteration();
```
On calcule la somme des itérations.
\note
C'est une variable de type `Arcane::VariableScalarInt32`. Il faut donc utiliser l'opérateur `()` pour récupérer sa valeur.
\warning
On modifie la variable `m_loop_sum` à chaque itération pour l'exemple. Dans les faits, utiliser l'affectation pour une variable de type `Arcane::VariableRefScalarT` peut s'avérer coûteuse.

____

```cpp
if (m_global_iteration() > options()->getNStep())
  subDomain()->timeLoopMng()->stopComputeLoop(true);
```
`m_global_iteration()` est une variable de la classe `CommonVariables`, dont hérite la classe `BasicModule` (dont hérite `ArcaneSayHelloObject` du fichier `SayHello_axl.h`). Cette variable contient l'itération actuellement en cours.
\note
C'est aussi une variable de type Arcane::VariableScalarInt32. Il faut donc utiliser l'opérateur `()` pour récupérer sa valeur.

`options()->getNStep()` permet de récupérer la valeur de l'option `nStep`. Pour définir cette valeur, il faut utiliser le jeu de données (fichier `.arc`) (voir prochaine section \ref arcanedoc_examples_simple_example_arc).

Enfin, `subDomain()->timeLoopMng()->stopComputeLoop(true)` permet de stopper la boucle en temps.

____

```cpp
ARCANE_REGISTER_MODULE_SAYHELLO(SayHelloModule);
```
Ceci est une macro permettant d'enregistrer le module dans %Arcane. Cette macro est définie dans le fichier `SayHello_axl.h`.

____

\remarks
Comme vous pouvez le constater, dans les trois fichiers au-dessus, il n'y a aucune mention de l'application `HelloWorld`. 
Les modules peuvent être placés et remplacés facilement dans d'autres applications.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_simple_example_struct
</span>
<span class="next_section_button">
\ref arcanedoc_examples_simple_example_arc
</span>
</div>