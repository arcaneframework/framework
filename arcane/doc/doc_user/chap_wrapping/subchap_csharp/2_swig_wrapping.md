# Extensions C# avec Swig {#arcanedoc_wrapping_csharp_swig}

[TOC]

Swig est un outil open source permettant d'interfacer du code C/C++
avec un autre langage. Dans le cas d'Arcane, on wrappe les classes C++
en C# pour qu'elles puissent être utilisée dans tout langage utilisant
'.Net'.

Il faut au moins la version 4.0 de swig.

Le wrapping se fait en décrivant dans un fichier avec l'extension .i
quels seront les fichiers wrappés et la manière de le faire. En
général, une classe C++ aura une classe C# de même nom. Par
convention, les méthodes C# commencent pas une majuscule. Pour
respecter cela, les méthodes des classes wrappées par Arcane sont
converties. Par exemple, la méthode C++ Arcane::ISubDomain::caseMng()
deviendra en C# la méthode `CaseMng()`.

La classe `Arcane::String` est convertie en la classe `string` de '.Net'.

\`A partir du fichier '.i', `swig` va générer un fichier `C++` et un
ensemble de fichiers `C#`. Les premiers doivent être compilés comme
un code `C++` normal sous forme de bibliothèque dynamique et les
seconds comme un projet `C#` classique. La communication entre le
`C++` et le `C#` se fait via un mécanisme de `.Net` appelé PInvoke
(pour `Platform Invoke`). Lors de l'exécution, la bibliothèque
compilée à partir du C++ doit être accessible. Pour cela, il faut
qu'elle soit dans le même répertoire que l'assembly `C#` ou alors
accessible via les variables d'environnement (LD_LIBRARY_PATH sous
Unix ou PATH sous Windows).

L'exemple 'eos/csharp' montre comment effectuer le wrapping et lancer
le code C#.

%Arcane utilise l'outil `swig` pour rendre accessible en `C#` les
différentes classes.

Ce document décrit comment le développeur peut ajouter ces propres
classes pour qu'elles soient accessibles en `C# `.

L'outil `swig` utilise un fichier d'extension `.i` pour décrire les
classes à wrapper.

Par exemple, on suppose qu'on a une interface C++ représentant
l'interface d'un service de calcul d'une équation d'état et qu'on
souhaite pouvoir implémenter ce service en `C#`. L'interface est
définie dans un fichier `IEquationOfState.h`:

```cpp
using namespace Arcane;

namespace EOS
{
//! Interface du service du modèle de calcul de l'équation d'état.
class IEquationOfState
{
 public:
  /** Destructeur de la classe */
  virtual ~IEquationOfState() = default;
  
 public:
  /*!
   *  Initialise l'équation d'état au groupe de mailles passé en argument
   *  et calcule la vitesse du son et l'énergie interne. 
   */
  virtual void initEOS(const CellGroup& group,
                       const VariableCellReal& pressure,
                       const VariableCellReal& adiabatic_cst,
                       const VariableCellReal& density,
                       VariableCellReal& internal_energy,
                       VariableCellReal& sound_speed
                       ) =0;
  /*!
   *  Applique l'équation d'état au groupe de mailles passé en argument
   *  et calcule la vitesse du son et la pression. 
   */
  virtual void applyEOS(const CellGroup & group,
                        const VariableCellReal& adiabatic_cst,
                        const VariableCellReal& density,
                        const VariableCellReal& internal_energy,
                        VariableCellReal& pressure,
                        VariableCellReal& sound_speed
                        ) = 0;
};

}
```

L'interface définit deux méthodes `initEOS` et `applyEOS` qu'on
souhaite rendre accessible en `C#`.

Pour cela, il faut définir un fichier `EOSCSharp.i` contenant le code
suivant :

```i
// 1ère partie
%module(directors="1") EOSCharp

%import core/ArcaneSwigCore.i

%typemap(csimports) SWIGTYPE
%{
using Arcane;
%}

%{
#include "ArcaneSwigUtils.h"
#include "arcane/ServiceFactory.h"
#include "arcane/ServiceBuilder.h"
#include "IEquationOfState.h"
using namespace Arcane;
%}

/*---------------------------------------------------------------------------*/
// 2ème partie
ARCANE_DECLARE_INTERFACE(EOS,IEquationOfState)

%include IEquationOfState.h

/*---------------------------------------------------------------------------*/
// 3ème partie
ARCANE_SWIG_DEFINE_SERVICE(EOS,IEquationOfState,
                           public abstract void InitEOS(CellGroup group,
                                                        VariableCellReal pressure,
                                                        VariableCellReal adiabatic_cst,
                                                        VariableCellReal density,
                                                        VariableCellReal internal_energy,
                                                        VariableCellReal sound_speed);
                           public abstract void ApplyEOS(CellGroup group,
                                                         VariableCellReal adiabatic_cst,
                                                         VariableCellReal density,
                                                         VariableCellReal internal_energy,
                                                         VariableCellReal pressure,
                                                         VariableCellReal sound_speed);
                           );
```

Ce fichier comporte trois parties:

1. La première partie commune à tous les wrappers qui décrit le nom du
   module et le code qui sera intégré au code généré. Il faut mettre
   dans cette partie tous les `.h` des classes qui seront wrappées par
   `swig`.
2. La deuxième partie qui indique explicitement les classes à
   wrapper. Si une classe `C++` doit être considérée comme une
   interface `C#`, il faut utiliser la macro
   `ARCANE_DECLARE_INTERFACE` pour le spécifier. Le premier paramètre
   est le nom du `namespace` et le second le nom de la classe. A noter
   que si on souhaite définir plusieurs interfaces, il faut toutes les
   déclarer avant de faire d'éventuels `%include`.
3. La troisième partie qui définit les interfaces qu'on souhaite
   étendre sous forme de service. On utilise pour cela la macro
   `ARCANE_SWIG_DEFINE_SERVICE`. Cette macro contient 3 arguments. Les
   deux premiers sont équivalents à ceux de la macro
   `ARCANE_DECLARE_INTERFACE`. Le dernier contient la signature `C#`
   des méthodes de l'interface wrappée. Elle n'est pas obligatoire en
   théorie mais permet de s'assurer que l'utilisateur implémente bien
   les méthodes de l'interface. En effet, via le mécanisme `swig` cela
   n'est pas obligatoirement garanti et si l'utilisateur ne surcharge
   pas les méthodes cela provoque une erreur lors de l'exécution. Swig
   va alors généré une exception de type `DirectorPureVirtualException`.

## Compilation du code généré par swig {#arcanedoc_wrapping_csharp_swig_build}

`swig` va générer deux types de fichiers :

- les fichiers 'C++' contenant le wrapping qui sera appelé directement
  depuis le `C#`.
- les fichiers `C#` qui contiennent les classes générées par `swig` et
  qui seront celles utilisées par le développeur.

Pour que le wrapping fonctionne, il faut donc à la fois compiler les
fichiers C++ sous la forme d'une bibliothèque dynamique et les
fichiers `C#` sous la forme d'une assembly.

\note Pour que le code `.Net` puisse appeler facilement du code natif
(C/C++), il faut que ce dernier soit accessible sous la forme d'une
bibliothèque dynamique. Il faut obligatoirement compiler le code
généré C++ généré par swig sous la forme d'une bibliothèque
dynamique. Pour être précis, il est possible de faire cela autrement
mais cela nécessite l'écriture de code dépendant du runtime (`mono` ou
`coreclr`) utilisé.

Si le développeur utilise `cmake`, alors il existe un package qui gère
à la fois la génération via swig et créé une cible pour le code C++
correspondant. Si notre fichier `.i` s'appelle `Wrapper.i` et que la
cible est `arcane_wrapper`, alors on peut utiliser le code suivant :

```cmake
set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(UseSWIG_MODULE_VERSION 2)
include(UseSWIG)

swig_add_library(arcane_wrapper
  TYPE SHARED
  LANGUAGE CSHARP
  SOURCES Wrapper.i
  )
```

\todo ajouter infos sur le package CMake ArcaneSwigUtils.cmake

Pour le code `C#`, il faut utiliser un fichier projet `C#` qui sera
compilé via la commande `dotnet build` ou `dotnet publish`.

   



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_wrapping_csharp_dotnet
</span>
<span class="next_section_button">
\ref arcanedoc_wrapping_csharp_casefunction
</span>
</div>
