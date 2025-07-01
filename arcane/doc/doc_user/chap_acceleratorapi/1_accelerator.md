# Utilisation des accélérateurs (GPU) {#arcanedoc_parallel_accelerator}

<!-- présente brièvement l'utilisation des accélérateurs dans %Arcane. -->

[TOC]

Dans ce chapître, on appellera accélérateur un co-processeur dedié
différent du processeur principal utilisé pour exécuter le code de
calcul. Dans la version actuelle de %Arcane, il s'agit d'accélérateurs
de type GPGPU.

L'API %Arcane pour gérer les accélérateurs s'inspire des bibliothèques
telles que [RAJA](https://github.com/LLNL/RAJA) ou
[Kokkos](https://github.com/kokkos/kokkos) mais se restreint aux
besoins spécifiques de %Arcane.

\note L'API accélérateur de %Arcane peut s'utiliser indépendamment des
mécanismes associés aux codes de calcul tels que les modules, le
maillage ou les services. Pour un exemple de fonctionnement autonome,
se reporter au chapître \ref arcanedoc_parallel_accelerator_standalone.

L'implémentation actuelle supporte uniquement comme accélérateur les
cartes graphiques NVidia (via CUDA) ou AMD (via ROCm).

L'API accélérateur de %Arcane répond aux objectifs suivants:

- unifier le comportement entre CPU séquentiel, CPU multi-thread et
  accélérateur.
- avoir un seul exécutable et pouvoir choisir dynamiquement où sera
  exécuté le code : CPU ou accélérateur (ou les deux à la fois).
- avoir un code source indépendant du compilateur et donc on n'utilise
  pas de mécanismes tels que les `#pragma` comme dans les normes
  OpenMP ou OpenACC.

\note Si on souhaite utiliser %Arcane à la fois sur GPU et sur CPU
pour l'environnement CUDA, il est fortement recommandé d'utiliser
`clang` comme compilateur au lieu de `nvcc` car ce dernier génère du
code moins performant sur la partie CPU. Cela est du à l'usage de
`std::function` pour encapsuler les lambdas utilisées dans %Arcane
(voir [New Compiler Features in CUDA 8](https://developer.nvidia.com/blog/new-compiler-features-cuda-8/#extended___host_____device___lambdas)
pour plus d'informations)

Le principe de fonctionnement est l'exécution de noyaux de calcul
déportés. Le code est exécuté par défaut sur le CPU (l'hôte) et
certaines parties du calcul sont déportés sur les accélérateurs. Ce
déport se fait via des appels spécifiques.

Pour utiliser les accélerateurs, il est nécessaire d'avoir compiler
%Arcane avec CUDA ou ROCm. Plus d'informations dans le chapitre 
\ref arcanedoc_build_install_build.

## Utilisation dans Arcane {#arcanedoc_parallel_accelerator_usage}

L'ensemble des types utilisés pour la gestion des accélérateurs est
dans l'espace de nom Arcane::Accelerator. Il y a deux composantes pour
gérer les accélérateurs :

- `arcane_accelerator_core` dont les fichiers d'en-tête sont inclus
  via `#include <arcane/accelerator/core>`. Cette composante comporte
  les classes indépendantes du type de l'accélérateur.

- `arcane_accelerator` dont les fichiers d'en-tête sont inclus
  via `#include <arcane/accelerator>`. Cette composante comporte
  les classes permettant de déporter des noyaux de calcul sur un
  l'accélérateur spécifique.
  
Les classes principales pour gérer les accélérateurs sont:

- \arcaneacc{IAcceleratorMng} qui permet d'accéder à
  l'environnement d'exécution par défaut.
- \arcaneacc{Runner} qui représente un environnement d'exécution
- \arcaneacc{RunQueue} qui représente une file d'exécution
- \arcaneacc{RunCommand} qui représente une commande (un
  noyau de calcul) associée à une file d'exécution.

Il existe deux possibilités pour utiliser les accélérateurs dans
%Arcane :

- via une instance de \arcaneacc{IAcceleratorMng} créé et
  initialisée par %Arcane au moment du lancement de
  l'exécutable (\ref arcanedoc_parallel_accelerator_module). C'est la
  méthode recommandée.
- via une instance de \arcaneacc{Runner} créée et
  initialisée manuellement (\ref arcanedoc_parallel_accelerator_runner).

Pour lancer un calcul sur accélérateur, il faut instancier une file
d'exécution. La classe \arcaneacc{RunQueue} gère une telle
file. La fonction \arcaneacc{makeQueue()} permet de créer une
telle file. Les files d'exécution peuvent être temporaires ou
persistantes mais ne peuvent pas être copiées. La méthode
\arcaneacc{makeQueueRef()} permet de créer une référence à
une file qui peut être copiée.

\note Par défaut la création de \arcaneacc{RunQueue} à partir
d'un \arcaneacc{Runner} n'est pas thread-safe pour des
raisons de performance. Si on veut pouvoir lancer plusieurs files
d'exécution à partir de la même instance de \arcaneacc{Runner} il faut appeler
la méthode \arcaneacc{Runner::setConcurrentQueueCreation(true)} avant

### Utilisation dans les modules {#arcanedoc_parallel_accelerator_module}

Il est possible pour tout module de récupérer une implémentation de
l'interface \arcaneacc{IAcceleratorMng} via la méthode
\arcane{AbstractModule::acceleratorMng()}. Le code suivant permet par
exemple d'utiliser les accélérateurs depuis un point d'entrée :

```cpp
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
    RunQueue* queue = acceleratorMng()->defaultQueue();
    // Boucle sur les mailles déportée sur accélérateur
    auto command1 = makeCommand(queue);
    command1 << RUNCOMMAND_ENUMERATE(Cell,vi,allCells()){
    };

    // Boucle classique 1D déportée sur accélérateur
    auto command2 = makeCommand(queue)
    command2 << RUNCOMMAND_LOOP1(iter,5){
    };
  }
};
```

### Instance spécifique de Runner {#arcanedoc_parallel_accelerator_runner}

Il est possible de créer plusieurs instances de l'objet
\arcaneacc{Runner}. 

Une instance de cette classe est associée à une politique d'exécution
dont les valeurs possibles sont données par l'énumération
\arcaneacc{eExecutionPolicy}. Par défaut, la politique
d'exécution est \arcaneacc{eExecutionPolicy::Sequential}, ce
qui signifie que les noyaux de calcul seront exécutés en séquentiel. 

\note Lorsqu'on créé une instance de \arcaneacc{Runner} sur
accélérateur, il est possible de spécifier un autre accélérateur que
l'accélérateur par défaut (si plusieurs sont disponibles). Cela
complique significativement la gestion de la mémoire. Le chapître \ref
arcanedoc_parallel_accelerator_multi explique comment gérer cela.

Il est aussi possible d'initialiser automatiquement une instance de cette
classe en fonction des arguments de la ligne de commande :

```cpp
#include "arcane/accelerator/RunQueue.h"
using namespace Arcane;
using namespace Arcane::Accelerator;
Runner runner;
ITraceMng* tm = ...;
IApplication* app = ...;
initializeRunner(runner,tm,app->acceleratorRuntimeInitialisationInfo());
```

## Compilation {#arcanedoc_parallel_accelerator_compilation}

%Arcane propose une intégration pour compiler avec le support des
accélérateurs via CMake. Ceux qui utilisent un autre système de
compilation doivent gérer aux même ce support.

Pour pouvoir utiliser des noyaux de calcul sur accélérateur, il faut
en général utiliser un compilateur spécifique. Par exemple, l'implémentation
actuelle de %Arcane via CUDA utilise le compilateur `nvcc` de NVIDIA
pour cela. Ce compilateur se charge de compiler la partie associée à
l'accélérateur. La partie associée au CPU est compilée avec le même
compilateur que le reste du code.

Il est nécessaire de spécifier dans le `CMakeLists.txt` qu'on souhaite
utiliser les accélérateurs ainsi que les fichiers qui seront compilés
pour les accélérateurs. Seuls les fichiers utilisant des commandes
(RUNCOMMAND_LOOP ou RUNCOMMAND_ENUMERATE) ont besoin d'être compilés
pour les accélérateurs. Pour cela, %Arcane définit les fonctions
CMake suivantes :

- **arcane_accelerator_enable()** qui doit être appelé vant les autres
  fonctions pour détecter l'environnement de compilation pour accélérateur
- **arcane_accelerator_add_source_files(file1.cc [file2.cc] ...)** pour
  indiquer les fichiers sources qui doivent être compilés sur accélérateurs
- **arcane_accelerator_add_to_target(mytarget)** pour indiquer que la
  cible `mytarget` a besoin de l'environnement accélérateur.

Si %Arcane est compilé en environnement CUDA, la variable CMake
`ARCANE_HAS_CUDA` est définie. Si %Arcane est compilé en environnement
HIP/ROCm, alors `ARCANE_HAS_HIP` est défini.

## Exécution {#arcanedoc_parallel_accelerator_exec}

Le choix de l'environnement d'exécution par défaut
(\arcaneacc{IAcceleratorMng::defaultRunner()}) est déterminé
par la ligne de commande :

- Si l'option `AcceleratorRuntime` est spécifiée, on utilise ce
  runtime. Actuellement les seules valeurs possibles sont `cuda` ou
  `hip`. Par exemple :
  ```sh
  MyExec -A,AcceleratorRuntime=cuda data.arc
  ```
- Sinon, si le multi-threading est activé via l'option `-T` (voir \ref
  arcanedoc_execution_launcher), alors les noyaux de calcul sont répartis sur
  plusieurs threads,
- Sinon, les noyaux de calcul sont exécutés en séquentiel.

## Noyaux de calcul (RunCommand) {#arcanedoc_parallel_accelerator_runcommand}

Une fois qu'on dispose d'une instance de \arcaneacc{RunQueue}, il est
possible de créér une commande qui pourra être déportée sur
accélérateur. Les commandes sont toujours des boucles qui peuvent être
de la forme suivante:

- Boucle classique de dimension 1 à 4. Cela se fait via les macros
  RUNCOMMAND_LOOP(), RUNCOMMAND_LOOP1(), RUNCOMMAND_LOOP2(),
  RUNCOMMAND_LOOP3() ou RUNCOMMAND_LOOP4().
- Boucle sur les entités du maillage. Cela se fait via la macro
  RUNCOMMAND_ENUMERATE().

Le chapître \ref arcanedoc_parallel_accelerator_lambda décrit la
syntaxe de ces boucles.

Le code suivant permet par exemple d'utiliser les accélérateurs depuis
un point d'entrée :

```cpp
// Fichiers à inclure tout le temps
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
    RunQueue queue = ...;

    // Boucle sur les mailles déportée sur accélérateur
    auto command1 = makeCommand(queue);
    command1 << RUNCOMMAND_ENUMERATE(Cell,vi,allCells()){
    };

    // Boucle classique 1D déportée sur accélérateur
    auto command2 = makeCommand(queue)
    command2 << RUNCOMMAND_LOOP1(iter,5){
    };
  }
};
```

### Utilisation des vues {#arcanedoc_parallel_accelerator_view}

Les accélérateurs ont en général leur propre mémoire qui est
différente de celle de l'hôte. Il est donc nécessaire de spécifier
comment seront utilisées les données pour gérer les éventuels
transferts entre les mémoires. Pour cela %Arcane fournit un mécanisme
appelé une vue qui permet de spécifier pour une variable ou un tableau
s'il va être utilisé en entrée, en sortie ou les deux.

\warning Une vue est un objet **TEMPORAIRE** et est toujours associée
à une commande (\arcaneacc{RunCommand}) et un conteneur
(Variable %Arcane ou tableau) et ne doit pas être utilisée lorsque la commande
associée est terminée ou le conteneur associé est modifié.

%Arcane propose des vues sur les variables (\arcane{VariableRef}) ou
sur la classe \arcane{NumArray} (La page \ref
arcanedoc_core_types_numarray décrit plus précisément l'utilisation de
cette classe).

Quel que soit le conteneur associé, la déclaration des vues est la
même et utilise les méthodes \arcaneacc{viewIn()},
\arcaneacc{viewOut()} ou \arcaneacc{viewInOut()}.

```cpp
// Pour avoir les NumArray
#include "arcane/utils/NumArray.h"

// Pour avoir les vues sur les variables
#include "arcane/accelerator/VariableViews.h"

// Pour avoir les vues sur les NumArray
#include "arcane/accelerator/NumArrayViews.h"

Arcane::Accelerator::RunCommand& command = ...;
// Tableaux 1D
Arcane::NumArray<Real,MDDim1> a;
Arcane::NumArray<Real,MDDim1> b;
Arcane::NumArray<Real,MDDim1> c;

// Variable 1D aux mailles
VariableCellReal var_c = ...;

// Vue en entrée (en lecture seule)
auto in_a = viewIn(command,a);

// Vue en entrée/sortie
auto inout_b = viewInOut(command,b);

// Vue en sortie (en écriture seule) sur la variable 'var_c'
auto out_c = viewOut(command,var_c);
```

### Gestion mémoire des données gérées par Arcane

Par défaut, %Arcane utilise l'allocateur retourné par
\arcane{MeshUtils::getDefaultDataAllocator()} pour le type
\arcane{NumArray} ainsi que toutes les variables
(\arcane{VariableRef}), les groupes d'entités
(\arcane{ItemGroup}) et les connectivités.

Lorsqu'on utilise les accélérateurs, %Arcane requiert que cet
allocateur alloue de la mémoire qui soit accessible à la fois sur
l'hôte et l'accélérateur. Cela signifie que les données
correspondantes à ces objets sont accessibles à la fois sur l'hôte
(CPU) et sur les accélérateurs. Pour cela, %Arcane utilise par défaut
la mémoire unifiée (\arccore{eMemoryResource::UnifiedMemory}).

Avec la mémoire unifiée, c'est l'accélérateur qui gère automatiquement
les éventules transferts mémoire entre l'accélérateur et l'hôte. Ces
transferts peuvent être coûteux en temps s'ils sont fréquents mais si
une donnée n'est utilisée que sur CPU ou que sur accélérateur, il n'y
aura pas de transferts mémoire et donc les performances ne seront pas
impactées.

A partir de la version 3.14.12 de %Arcane, il est possible de changer
la ressoure mémoire utilisée par défaut via la variable
d'environnement `ARCANE_DEFAULT_DATA_MEMORY_RESOURCE`. Sur les
accélérateurs où la mémoire \arccore{eMemoryResource::Device} est
accessible directement depuis l'hôte (par exemple MI250X, MI300A,
GH200), cela permet d'éviter les transferts que peut provoquer la
mémoire unifiée.

Dans tous les cas, il est possible de spécifier un allocateur
spécifique pour \arccore{UniqueArray} et \arcane{NumArray} via les
méthodes \arcane{MemoryUtils::getAllocator()} ou
\arcane{MemoryUtils::getAllocationOptions()}.

%Arcane fournit des mécanismes permettant de donner des informations
permettant d'optimiser la gestion de cette mémoire. Ces mécanismes
sont dépendants du type de l'accélérateur et peuvent ne pas être
disponible partout. Ils sont accessibles via la méthode
\arcaneacc{Runner::setMemoryAdvice()}.

A partir de la version 3.10 de %Arcane et avec les accélérateurs
NVIDIA, %Arcane propose des fonctionnalités pour détecter les
transferts mémoire entre le CPU et l'accélérateur. La page \ref
arcanedoc_debug_perf_cupti décrit ce fonctionnement.

### Exemple d'utilisation d'une boucle complexe {#arcanedoc_parallel_accelerator_complexloop}

L'exemple suivant montre comment modifier l'intervalle d'itération
pour ne pas partir de zéro :

```cpp
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
```

### Utilisation des lambda {#arcanedoc_parallel_accelerator_lambda}

Quelle que soit la macro (RUNCOMMAND_ENUMERATE(),
RUNCOMMAND_LOOP(), ...) utilisée pour la boucle, le code qui suit doit
être une une fonction [lambda du
C++11](https://en.cppreference.com/w/cpp/language/lambda). C'est cette
fonction lambda qui sera éventuellement déportée sur accélérateur.

%Arcane utilise l'opérateur `operator<<` pour "envoyer" la boucle sur
une commande (\arcaneacc{RunCommand}) ce qui permet d'écrire le code
de manière similaire à celui d'une boucle C++ classique (ou une boucle
ENUMERATE_() dans le cas des entités du maillage) avec les quelques
modifications suivantes :

- les accolades (`{` et `}`) sont obligatoires
- il faut ajouter un `;` après la dernière accolade.
- le corps d'une lambda est une fonction et pas une boucle. Par
  conséquent, il n'est pas possible d'utiliser les mots clés tels que
  `continue` ou `break`. Le mot clé `return` est disponible et donc
  aura le même effet que `continue` dans une boucle.

Par exemple :

```cpp
Arcane::Accelerator::RunCommand& command = ...
// Boucle 1D de 'nb_value' avec 'iter' l'itérateur
command << RUNCOMMAND_LOOP1(iter,nb_value)
{
  // Code exécuté sur accélérateur
};
```

```cpp
Arcane::Accelerator::RunCommand& command = ...
// Boucle sur les mailles du groupe 'my_group' avec 'cid' l'indice de
// la maille courante (de type Arcane::CellLocalId)
command << RUNCOMMAND_ENUMERATE(Cell,icell,my_group)
{
  // Code exécuté sur accélérateur
};
```

Lorsque'un noyau de calcul est déporté sur accélérateur, il ne faut
pas accéder à la mémoire associée aux vues depuis une autre partie du
code pendant l'exécution sous peine de plantage. En général cela ne
peut se produire que lorsque les \arcaneacc{RunQueue} sont
asynchrones. Par exemple :

```cpp
#include "arcane/accelerator/Views.h"
using namespace Arcane::Accelerator;
Arcane::Accelerator::RunQueue& queue = ...;
queue.setAsync(true);
Arcane::NumArray<Real,MDDim1> a;
Arcane::NumArray<Real,MDDim1> b;

Arcane::Accelerator::RunCommand& command = makeCommand(queue);
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
```

### Limitation des lambda C++ sur accélérateurs {#arcanedoc_parallel_accelerator_limitlambda}

Les mécanismes de compilation et la gestion mémoire sur accélérateurs
font qu'il y a des restrictions sur l'utilisation des lambda
classiques du C++

#### Appel d'autres fonctions dans les lambda {#arcanedoc_parallel_accelerator_callslambda}

Dans une lambda prévue pour être déportée sur accélérateur, on ne peut
appeler que :

- des méthodes de classe qui sont **publiques**
- qui fonctions qui sont `inline`
- qui fonctions ou méthodes qui ont l'attribut ARCCORE_HOST_DEVICE ou
  ARCCORE_DEVICE ou des méthodes `constexpr`

Il n'est pas possible d'appeler des fonctions externes qui sont
définies dans d'autres unités de compilation (par exemple d'autres
bibliothèques)

#### Utilisation des champs d'une instance de classe {#arcanedoc_parallel_accelerator_classinstance}

Il ne faut pas utiliser dans les lambdas une référence à un champ
d'une classe car ce dernier est capturé par référence. Cela provoquera
un plantage par accès mémoire invalide sur accélérateur. Pour éviter
ce problème, il suffit de déclarer localement à la fonction une copie
de la valeur de l'instance de classe qu'on souhaite utiliser. Dans
l'exemple suivant la fonction `f1()` provoquera un plantage alors que
`f2()` fonctionnera bien.

```cpp
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
  Arcane::NumArray<int,MDDim1> a(100);
  auto out_a = viewIn(command,a);
  command << RUNCOMMAND_LOOP1(iter,100){
    out_a(iter) = my_value+5; // BAD !!
  };
}
void A::f2()
{
  Arcane::Accelerator::RunCommand& command = ...
  Arcane::NumArray<int,MDDim1> a(100);
  auto out_a = viewIn(command,a);
  int v = my_value;
  command << RUNCOMMAND_LOOP1(iter,100){
    out_a(iter) = v+5; // GOOD !!
  };
}
```

## Utilisation du mécanisme d'échange de message

A partir de la version 3.10, %Arcane supporte les bibliothèques MPI
"Accelerator Aware". Dans ce cas, le buffer utilisé pour les
synchronisations des variables est alloué directement sur
l'accélérateur. Si une variable est utilisée sur accélérateur cela
permet donc d'éviter des recopies inutiles entre l'hôte et
l'accélérateur. Le mode échange de message en mémoire partagée
supporte aussi ce mécanisme.

En cas de problèmes, il est possible de désactiver ce support en
positionnant la variable d'environnement
`ARCANE_DISABLE_ACCELERATOR_AWARE_MESSAGE_PASSING` à une valeur non
nulle.

## Gestion des multi-accélérateurs {#arcanedoc_parallel_accelerator_multi}

%Arcane associe lors de la création d'un sous-domaine une instance de
\arcaneacc{Runner} (accessible via
\arcane{ISubDomain::acceleratorMng()}).
Lorsqu'une machine dispose de plusieurs accélérateurs, %Arcane choisi
par défaut le premier qui est retourné dans les accélérateurs
disponibles. Il est possible de changer ce comportement en
positionnant la variable d'environnement
`ARCANE_ACCELERATOR_PARALLELMNG_RANK_FOR_DEVICE` à une valeur
strictement positive indiquant le modulo entre le rang de sous-domaine
(retourné par \arcane{IParallelMng::commRank()} de
\arcane{ISubDomain::parallelMng()}) et l'index de l'accélérateur dans
la liste des accélérateurs. Par exemple si cette variable
d'environnement vaut 8, alors le sous-domaine de rang N sera associé à
l'accélérateur d'index \a (N % 8). Pour que ce mécanisme fonctionne,
la valeur de cette variable d'environnemetn doit donc être inférieure
au nombre d'accélérateurs disponibles sur la machine.

### Gestion mémoire

Lorsque plusieurs accélérateurs sont disponibles sur une même
machine, il existe en général un accélérateur "courant" pour chaque
thread (par exemple avec CUDA il est possible de le récupérer par la méthode
`cudaGetDevice()` et on peut le changer par la méthode
`cudaSetDevice()`). Lorsqu'on alloue de la mémoire sur
accélérateur, c'est sur cet accélérateur "courant" et cette mémoire ne
sera pas disponible sur d'autres accélérateurs. Une instance de
\arcaneacc{RunQueue} est associée à un accélérateur donné et
il faut donc s'assurer que les zones mémoires utilisées par une
commande sont bien accessibles. Si ce n'est pas le cas cela produira
une erreur lors de l'exécution (Par exemple, avec CUDA, il s'agit de l'erreur 400
dont le message est "invalid resource handle").

Si l'accélérateur "courant" a été modifié par exemple lors de l'appel
à une bibliothèque externe il est possible de le changer en appelant
la méthode \arcaneacc{Runner::setAsCurrentDevice()}.

## Gestion des connectivités et des informations sur les entités

L'accès aux connectivités du maillage se fait différemment sur
accélérateur que sur le CPU pour des raisons de performance. Il n'est
notamment pas possible d'utiliser les entités classiques
(\arcane{Cell},\arcane{Node}, ...). A la place il faut utiliser les
indentifiants locaux tels que \arcane{CellLocalId} ou
\arcane{NodeLocalId}.

La classe \arcane{UnstructuredMeshConnectivityView} permet d'accéder
aux informations de connectivité. Il est possible de définir une
instance de cette classe et de la conserver au cours du calcul. Pour
initialiser l'instance, il faut appeler la méthode
\arcane{UnstructuredMeshConnectivityView::setMesh()}.

\warning Comme toutes les vues, l'instance est invalidé lorsque le
maillage évolue. Il faut donc à nouveau appeler
\arcane{UnstructuredMeshConnectivityView::setMesh()} après une
modification du maillage.

Pour accéder aux informations génériques des entités, comme le type ou
le propriétaire, il faut utiliser la vue
\arcane{ItemGenericInfoListView}.

L'exemple suivant montre comment accéder aux noeuds des mailles et aux
informations des mailles. Il parcourt l'ensemble des mailles et calcule
le barycentre pour celles qui sont dans notre sous-domaine et qui sont
des hexaèdres.

\snippet accelerator/SimpleHydroAcceleratorService.cc AcceleratorConnectivity

## Opérations atomiques

La méthode \arcaneacc{doAtomic} permet d'effectuer des opérations
atomiques. Les types d'opérations supportées sont définies par
l'énumération \arcaneacc{eAtomicOperation}. Par exemple:

\snippet AtomicUnitTest.cc SampleAtomicAdd

## Algorithmes avancés: Réductions, Scan, Filtrage, Partitionnement et Tri

%Arcane propose plusieurs classes permettant d'effectuer des
algorithmes plus avancés. Sur accélérateur, ces algorithmes utilisent
en général les bibliothèques proposées par le constructeur
([CUB](https://nvidia.github.io/cccl/cub/index.html) pour NVIDIA et
[rocprim](https://rocm.docs.amd.com/projects/rocPRIM/en/develop/reference/reference.html)
pour AMD). Les algorithmes proposés par %Arcane possèdent donc les
mêmes limitations que l'implémentation constructeur sous-jacente.

Les classes disponibles sont:

- \arcaneacc{GenericFilterer} pour filtrer les éléments d'un tableau.
- \arcaneacc{GenericScanner} pour effectuer des algorithmes de scan inclusifs ou exclusifs (voir
  [Algorithmes de Scan](https://en.wikipedia.org/wiki/Prefix_sum) sur wikipedia)
- \arcaneacc{GenericSorter} pour trier les éléments d'une liste
- \arcaneacc{GenericPartitioner} pour partitionner les éléments d'une liste
- \arcaneacc{GenericReducer} pour effectuer des réduction. Il existe
  aussi d'autres manières de réaliser des réductions qui sont
  décrites dans la page (\ref arcanedoc_acceleratorapi_reduction)

## Mode Autonome accélérateur {#arcanedoc_parallel_accelerator_standalone}

Il est possible d'utiliser le mode accélérateur de %Arcane sans le
support des objets de haut niveau tel que les maillages ou les
sous-domaines.

Dans ce mode, il est possible d'utiliser l'API accélérateur de %Arcane
directement depuis la fonction `main()` par exemple. Pour utiliser ce
mode, il suffit d'utiliser la méthode de classe
\arcane{ArcaneLauncher::createStandaloneAcceleratorMng()} après avoir
initialiser %Arcane :

```cpp
Arcane::ArcaneLauncher::init(Arcane::CommandLineArguments(&argc, &argv));
Arcane::StandaloneAcceleratorMng launcher(Arcane::ArcaneLauncher::createStandaloneAcceleratorMng());
```

L'instance `launcher` doit rester valide tant qu'on souhaite utiliser
l'API accélérateur. Il est donc préférable de la définir dans le
`main()` du code. La classe \arcane{StandaloneAcceleratorMng} utilise
une sématique par référence. Il est donc possible de conserver une
référence vers l'instance n'importe où dans le code si nécessaire.

L'exemple 'standalone_accelerator' montre une telle
utilisation. Par exemple, le code suivant permet de déporter sur
accélérateur la somme de deux tableaux `a` et `b` dans un tabeau `c`.

\snippet standalone_accelerator/main.cc StandaloneAcceleratorFull

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_acceleratorapi
</span>
<span class="next_section_button">
\ref arcanedoc_accelerator_materials
</span>
</div>
