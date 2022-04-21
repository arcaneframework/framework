Utilisation des accélérateurs {#arcanedoc_accelerator}
===================

[TOC]

\warning L'API d'utilisation des accélérateurs est en cours de
développement et n'est pas figée. De plus il ne faut pas utiliser les
classes autres que celles documentées dans cette page.

On appelle accélérateur un co-processeur dedié différent du processeur
principal utilisé pour exécuté le code de calcul.

L'API %Arcane pour gérer les accélérateurs s'inspire des bibliothèques
telles que [RAJA](https://github.com/LLNL/RAJA) ou
[Kokkos](https://github.com/kokkos/kokkos) mais se restreint aux
besoins spécifiques de %Arcane.

\note L'implémentation actuelle supporte uniquement comme accélérateur les
cartes graphiques NVidia.

Le but de L'API est:

- pouvoir choisir dynamiquement où sera exécuté le code: CPU ou
  accélérateur (ou les deux à la fois)
- avoir un code source indépendant du compilateur

Le principe de fonctionnement est l'exécution de noyaux de calcul
déportés. Le code est exécuté par défaut sur le CPU (l'hôte) et
certains parties du calcul sont déportés sur les accélérateurs. Ce
déport se fait via des appels spécifiques.

## Utilisation dans Arcane

L´ensemble des types utilisés pour la gestion des accélérateurs est
dans l'espace de nom Arcane::Accelerator. Les classes principales
sont:

- Arcane::Accelerator::IAcceleratorMng qui permet d'accéder à
  l'environnement d'exécution par défaut.
- Arcane::Accelerator::Runner qui représente un environnement d'exécution
- Arcane::Accelerator::RunQueue qui représente une file d'exécution
- Arcane::Accelerator::RunCommand qui représente une commande (un
  noyau de calcul) associée à une file d'exécution.

Il existe deux possibilités pour utiliser les accélérateurs dans
Arcane:
- via une instance de Arcane::Accelerator::IAcceleratorMng créé et
  initialisée par %Arcane au moment du lancement de l'exécutable.
- via une instgance de Arcane::Accelerator::Runner créée et
  initialisée manuellement.

### Utilisation dans les modules

il est possible pour tout module de récupérer une implémentation de
l'interface Arcane::Accelerator::IAcceleratorMng via la méthode
Arcane::AbstractModule::acceleratorMng(). Le code suivant permet par
exemple d'utiliser les accélérateurs depuis un point d'entrée:

~~~{.cpp}
// Fichier à include tout le temps
#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/RunQueue.h"

// Fichier à inclure pour avoir RUNCOMMAND_ENUMERATE
#include "arcane/accelerator/RunCommandEnumerate.h"

// Fichier à inclure pour avoir RUNCOMMAND_LOOP
#include "arcane/accelerator/RunCommandLoop.h"

using namespace Arcane;
using namespace Arcane::Accelerator;

class MyModule
: public Arcane::BasicModule
{
 public:
  void myEntryPoint()
  {
    // Boucle sur les mailles déportée sur accélérateur
    auto command1 = makeCommand(acceleratorMng()->defaultQueue());
    command1 << RUNCOMMAND_ENUMERATE(Cell,vi,allCells()){
    };

    // Boucle classique 1D déportée sur accélérateur
    auto command2 = makeCommand(acceleratorMng()->defaultQueue());
    command2 << RUNCOMMAND_LOOP1(iter,5){
    };
  }
};
~~~

### Utilisation via une instance spécifique de 'Runner'

L'objet principal est la classe Arcane::Accelerator::Runner. Il est
possible de créér plusieurs instances de cet objet Arcane::Accelerator::Runner.

\note Actuellement, les méthodes de Arcane::Accelerator::Runner ne
sont pas thread-safe.

Une instance de cette classe est associée à une politique d'exécution
dont les valeurs possibles sont données par l'énumération
Arcane::Accelerator::eExecutionPolicy. Par défaut, la politique
d'exécution est Arcane::Accelerator::eExecutionPolicy::Sequential, ce
qui signifie que les noyaux de calcul seront exécutés en séquentiel. 

Il est aussi possible d'initialiser automatiquement une instance de cette
classe en fonction des arguments de la ligne de commande:

~~~{.cpp}
#include "arcane/accelerator/RunQueue.h"
using namespace Arcane;
using namespace Arcane::Accelerator;
Runner runner;
ITraceMng* tm = ...;
IApplication* app = ...;
initializeRunner(runner,tm,app->acceleratorRuntimeInitialisationInfo());
~~~

Pour lancer un calcul sur accélérateur, il faut instancier une file
d'exécution. La classe Arcane::Accelerator::RunQueue gère une telle
file. La fonction Arcane::Accelerator::makeQueue() permet de créer une
telle file. Les files d'exécution peuvent être temporaires ou
persistantes mais ne peuvent pas être copiées. La méthode
Arcane::Accelerator::makeQueueRef() permet de créer une référence à
une file qui peut être copiée.

## Exemple d'utilisation d'une boucle complexe

~~~{.cpp}
using namespace Arcane;
using namespace Arcane::Accelerator;
{
  Arcane::Accelerator::Runner runner = ...;
  auto queue = makeQueue(runner);
  auto command = makeCommand(queue);
  auto out_t1 = viewOut(command,t1);
  Int64 base = 300;
  Int64 s1 = 400;
  auto b = makeLoopRanges({base,s1},n2,n3,n4);
  command << RUNCOMMAND_LOOP(iter,b)
  {
    auto [i, j, k, l] = iter();
    out_t1(i,j,k,l) = _getValue(i,j,k,l);
  };
}
~~~

## Compilation

%Arcane propose une intégration pour compiler avec le support des
accélérateurs via CMake. Ceux qui utilisent un autre système de
compilation doivent gérer aux même ce support.

Pour pouvoir utiliser des noyaux de calcul sur accélérateur, il faut
en général utiliser un compilateur spécifique. L'implémentation
actuelle de %Arcane via CUDA utilise le compilateur `nvcc` de NVIDIA
pour cela. Ce compilateur se charge de compiler la partie associée à
l'accélérateur. La partie associée au CPU est compilée avec le même
compilateur que le reste du code.

Il est nécessaire de spécifier dans le `CMakeLists.txt` qu'on souhaite
utiliser les accélérateurs ainsi que les fichiers qui seront compilés
pour les accélérateurs. Seuls les fichiers utilisant des commandes
(RUNCOMMAND_LOOP ou RUNCOMMAND_ENUMERATE) ont besoin d'être compilés
pour les accélérateurs. Pour cela, %Arcane définit les fonctions
CMake suivantes:

- **arcane_accelerator_enable()** qui doit être appelé vant les autres
  fonctions pour détecter l'environnement de compilation pour accélérateur
- **arcane_accelerator_add_source_files(file1.cc [file2.cc] ...)** pour
  indiquer les fichiers sources qui doivent être compilés sur accélérateurs
- **arcane_accelerator_add_to_target(mytarget)** pour indiquer que la
  cible `mytarget` a besoin de l'environnement accélérateur.

Si %Arcane est compilé en environnement CUDA, la variable CMake
`ARCANE_HAS_CUDA` est définie.

## Exécution

Le choix de l'environnement d'exécution par défaut
(Arcane::Accelerator::IAcceleratorMng::defaultRunner()) est déterminé
par la ligne de commande:

- Si l'option `AcceleratorRuntime` est spécifiée, on utilise ce
  runtime. Actuellement la seule valeur possible est `cuda`. Par
  exemple:
  ~~~{.sh}
  MyExec -A,AcceleratorRuntime=cuda data.arc
  ~~~
- Sinon, si le multi-threading est activé via l'option `-T` (voir \ref
  arcanedoc_launcher), alors les noyaux de calcul sont répartis sur
  plusieurs threads,
- Sinon, les noyaux de calcul sont exécutés en séquentiel.

## Utilisation des vues

Les accélérateurs ont en général leur propre mémoire qui est
différente de celle de l'hôte. Il est donc nécessaire de spécifier
comment seront utiliser les données pour gérer les éventuels
transferts entre les mémoire. Pour cela %Arcane fournit un mécanisme
appelé une vue qui permet de spécifier pour une variable ou un tableau
s'il va être utilisé en entrée, en sortie ou les deux.

\warning Une vue est un objet **TEMPORAIRE** et est toujours associée
à une commande (Arcane::Accelerator::RunCommand) et un conteneur
(Variable %Arcane ou tableau) et ne doit pas être utilisée lorsque la commande
associée est terminée ou le conteneur associé est modifié.

%Arcane propose des vues sur les variables (Arcane::IVariable) ou sur la classe Arcane::NumArray.

Quel que soit le conteneur associé, la déclaration des vues est la
même:

~~~{.cpp}
#include "arcane/utils/NumArray.h"
// Pour avoir les vues sur les variables
#include "arcane/accelerator/VariableViews.h"
// Pour avoir les vues sur les NumArray
#include "arcane/accelerator/NumArrayViews.h"
using namespace Arcane;
using namespace Arcane::Accelerator;
RunCommand& command = ...;
Arcane::NumArray<Real,1> a;
Arcane::NumArray<Real,1> b;
Arcane::NumArray<Real,1> c;
VariableCellReal var_c = ...;
auto in_a = viewIn(command,a); // Vue en entrée
auto inout_b = viewInOut(command,b); // Vue en entrée/sortie
auto out_c = viewOut(command,var_c); // Vue en sortie.
~~~

## Utilisation des lambda

Actuellement il est possible de déporter deux types de boucles sur
accélérateurs:
- les boucles sur les entités du maillage via la macro
  RUNCOMMAND_ENUMERATE()
- les boucles classiques sur des tableaux via la commande
  RUNCOMMAND_LOOP().

Ces deux macros permettent de définir après un bout de code qui est
une fonction lambda du C++11 (TODO: ajouter référence) et qui sera
déporté sur accélérateur. Ces macros s'utilisent via l'opérateur
'operator<<' sur une commande (Arcane::Accelerator::RunCommand). Le
code après la macro est un code identique à celui d'une boucle C++
classique avec les modifications suivantes:

- les accolades sont obligatoires
- il faut ajouter un `;` après la dernière accolade.
- le corps d'une lambda est une fonction et pas une boucle. Par
  conséquent, il n'est pas possible d'utiliser les mot clés tels que
  `continue` ou `break`. Le mot clé `return` est disponible et donc
  aura le même effet que `continue` dans une boucle.

Par exemple:

~~~{.cpp}
Arcane::Accelerator::RunCommand& command = ...
// Boucle 1D de 'nb_value' avec 'iter' l'itérateur
command << RUNCOMMAND_LOOP1(iter,nb_value)
{
  // Code exécuté sur accélérateur
};
~~~

~~~{.cpp}
Arcane::Accelerator::RunCommand& command = ...
// Boucle sur les mailles du groupe 'my_group' avec 'cid' l'indice de
// la maille courante (de type Arcane::CellLocalId)
command << RUNCOMMAND_ENUMERATE(Cell,icell,my_group)
{
  // Code exécuté sur accélérateur
};
~~~

Lorsque'un noyau de calcul est déporté sur accélérateur, il ne faut
pas accéder à la mémoire associée aux vues pendant l'exécution sous
peine de plantage. En général cela ne peut se produire que lorsque les
Arcane::Accelerator::RunQueue sont asynchrones. Par exemple:

~~~{.cpp}
#include "arcane/accelerator/Views.h"
using namespace Arcane::Accelerator;
RunQueue& queue = ...;
queue.setAsync(true);
Arcane::NumArray<Real,1> a;
Arcane::NumArray<Real,1> b;

RunCommand& command = makeCommand(queue);
auto in_a = viewIn(command,a);
auto out_b = viewOut(command,b);
// Copie A dans B
command << RUNCOMMAND_LOOP1(iter,nb_value)
{
  auto [i] = iter();
  out_b(i) = in_a(i);
};
// La commande est en cours d'exécution tant que la méthode barrier()
// n'a pas été appelée

// ICI il NE FAUT PAS utiliser 'a' ou 'b' ou 'in_a' ou 'out_b'

queue.barrier();

// ICI on peut utiliser 'a' ou 'b' (MAIS PAS 'in_a' ou 'out_b' car la
// commande est terminée)
~~~

## Limitation des lambda C++ sur accélérateurs

Les mécanismes de compilation et la gestion mémoire sur accélérateurs
font qu'il y a des restrictions sur l'utilisation des lambda
classiques du C++

### Appel d'autres fonctions dans les lambda

Dans une lambda prévue pour être déportée sur accélérateur, on ne peut
appeler que:

- des méthodes de classe qui sont **publiques**
- qui fonctions qui sont `inline`
- qui fonctions ou méthodes qui ont l'attribut ARCCORE_HOST_DEVICE ou
  ARCCORE_DEVICE ou des méthodes `constexpr`

Il n'est pas possible d'appeler des fonctions externes qui sont
définies dans d'autres unités de compilation (par exemple d'autres
bibliothèques)

### Utilisation des champs d'une instance de classe

Il ne faut pas utiliser dans les lambdas une référence à un champ
d'une classe car ce dernier est capturé par référence. Cela provoquera
un plantage par accès mémoire invalide sur accélérateur. Pour éviter
ce problème, il suffit de déclarer localement à la fonction une copie
de la valeur de l'instance de classe qu'on souhaite utiliser. Dans
l'exemple suivant la fonction `f1()` provoquera un plantage alors que
`f2()` fonctionnera bien.

~~~{.cpp}
class A
{
public:
 void f1();
 void f2();
 int my_value;
};
void A::f1()
{
  Arcane::Accelerator::RunCommand& command = ...
  Arcane::NumArray<int,1> a(100);
  auto out_a = viewIn(command,a);
  command << RUNCOMMAND_LOOP1(iter,100){
    out_a(iter) = my_value+5; // BAD !!
  };
}
void A::f2()
{
  Arcane::Accelerator::RunCommand& command = ...
  Arcane::NumArray<int,1> a(100);
  auto out_a = viewIn(command,a);
  int v = my_value;
  command << RUNCOMMAND_LOOP1(iter,100){
    out_a(iter) = v+5; // GOOD !!
  };
}
~~~

## Mode Autonome

Il est possible d'utiliser le mode accélérateur de %Arcane sans le
support des objets de haut niveau tel que les maillages ou les
sous-domaines. L'exemple 'standalone_accelerator' montre une telle
utilisation. Par exemple, le code suivant permet de déporter sur
accélérateur la somme de deux tableaux `a` et `b` dans un tabeau `c`.

~~~{.cpp}
int
main(int argc,char* argv[])
{
  using namespace Arcane;

  // Teste la somme de deux tableaux 'a' et 'b' dans un tableau 'c'.
  try{
    ArcaneLauncher::init(CommandLineArguments(&argc,&argv));
    StandaloneAcceleratorMng launcher(ArcaneLauncher::createStandaloneAcceleratorMng());
    IAcceleratorMng* acc_mng = launcher.acceleratorMng();

    constexpr int nb_value = 10000;


    // Définit 2 tableaux 'a' et 'b' et effectue leur initialisation.
    NumArray<Int64,1> a(nb_value);
    NumArray<Int64,1> b(nb_value);
    for( int i=0; i<nb_value; ++i ){
      a.s(i) = i+2;
      b.s(i) = i+3;
    }

    // Defínit le tableau 'c' qui contiendra la somme de 'a' et 'b'
    NumArray<Int64,1> c(nb_value);

    // Noyau de calcul déporté sur accélérateur.
    {
      auto command = makeCommand(acc_mng->defaultQueue());
      // Indique que 'a' et 'b' seront en entrée et 'c' en sortie

      auto in_a = viewIn(command,a);
      auto in_b = viewIn(command,b);
      auto out_c = viewOut(command,c);
      command << RUNCOMMAND_LOOP1(iter,nb_value)
      {
        out_c(iter) = in_a(iter) + in_b(iter);
      };
    }
  }
  catch(const Exception& ex){
    std::cerr << "EXCEPTION: " << ex << "\n";
    return 1;
  }
  return 0;
}
~~~


## TODO

TODO: expliquer utilisation des NumArray.

TODO: expliquer l'utilisation des connectivités et pourquoi on ne peut
pas accéder aux entités classiques (Cell,Node, ...) sur accélérateur.
