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

L'interface principale est Arcane::Accelerator::IAcceleratorMng. Une
implémentation de cette interface est créé lors de l'initialisation et
est disponible pour les modules via la méthode
Arcane::AbstractModule::acceleratorMng().

L'objet principal est la classe Arcane::Accelerator::Runner. Il est
possible de créér plusieurs instances de cet objet.

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

## Exemple d'utilisation

~~~{.cpp}
using namespace Arcane;
using namespace Arcane::Accelerator;
{
  Runner runner = ...;
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

- **arcane_accelerator_enable()** qui doit être appeler vant les autres
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

## TODO

TODO: expliquer la capture des lambda avec le 'this'

TODO: expliquer mémoire unifiée ne pas utiliser le CPU en même temps que le GPU

TODO: expliquer utilisation des NumArray.

TODO: expliquer utilisation des vues
