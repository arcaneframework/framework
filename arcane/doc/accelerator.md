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
dans l'espace de nom Arcane::Accelerator

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

TODO Exemple itération

~~~{.cpp}
using namespace Arcane;
using namespace Arcane::Accelerator;
{
  auto queue = makeQueue(m_runner);
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


TODO: utilisation via CMake.

TODO: expliquer la capture des lambda avec le 'this'

TODO: expliquer mémoire unifiée ne pas utiliser le CPU en même temps que le GPU

TODO: expliquer utilisation des NumArray.

TODO: expliquer utilisation des vues
