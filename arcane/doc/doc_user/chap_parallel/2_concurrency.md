# Concurrence et multi-threading {#arcanedoc_parallel_concurrency}

[TOC]

<!-- décrite la notion de concurrence et la parallélisation au niveau des boucles -->

La notion de concurrence est implémentée dans %Arcane via la notion de tâche.

Cette notion de tâche permet l'exécution concurrente de plusieurs
opérations via les threads.

Cette notion est complémentaire de la notion de décomposition de
domaine utilisée par le Arcane::IParallelMng. Il est donc tout à fait
possible de mélanger décomposition de domaine et les thread.

\warning Néanmoins, si l'implémentation de Arcane::IParallelMng se fait via
MPI, il est déconseillé de faire des appels au Arcane::IParallelMng lorsque
des tâches se déroulent de manière concurrente, par exemle dans les
boucles parallélisées. La plupart des implémentations MPI ne sont pas
très performantes dans ce mode et certaines ne le supporte que
partiellement.

Pour utiliser les tâches, il faut inclure le fichier suivant:

```cpp
#include "arcane/Concurrency.h"
```

Il existe deux mécanismes pour utiliser les tâches:

1. Implicitement via la notion de boucle parallèle
2. explicitement en créant les tâches directement
 
La première solution est la plus simple et doit être envisagée en priorité.

## Activation {#arcanedoc_parallel_concurrency_activation}

Par défaut, le support de la concurrence est désactivé. L'activation
se fait **avant** le lancement du code, en spécifiant le
nombre de tâches pouvant s'exécuter de manière concurrentes lors de la
ligne de commande (se reporter à la page \ref arcanedoc_execution_launcher pour
savoir comment faire cela).
 
Il est possible de savoir dans le code si la concurrence est active en
appelant la méthode Arcane::TaskFactory::isActive().

Il n'est pas possible d'activer la concurrence pendant l'exécution.

## Boucles parallèles {#arcanedoc_parallel_concurrency_parallel_for}

Il existe deux formes de boucles parallèles. La première forme s'applique
sur les boucles classiques, la seconde sur les groupes d'entités.

Le mécanisme de fonctionnement est similaire aux directives
`omp parallel for` de OpenMp.

\warning L'utilisateur de ce mécanisme doit s'assurer que la boucle
peut être correctement parallélisée sans qu'il y ait d'effets de
bord. Notamment, cela inclut (mais ne se limite pas) la garantie que
les itérations de la boucle sont indépendantes, qu'il n'y a pas
d'opérations de sortie de boucle (return, break). 

La première forme est pour paralléliser la boucle séquentielle suivante:

```cpp
void func()
{
  for( Integer i=0; i<n; ++i )
    p[i] = (gamma[i]-1) * rho[i] * e[i];
}
```

La parallélisation se fait comme suit: il faut d'abord écrire une
classe fonctor qui représente l'opération que l'on souhaite effectuée
sur un interval d'itération. Ensuite, il faut utiliser l'opération
arcaneParallelFor() en spécifiant ce fonctor en argument comme suit:

```cpp
class Func
{
  public:
   void exec(Integer begin,Integer size)
   {
     for( Integer i=begin; i<(begin+size); ++i )
       p[i] = (gamma[i]-1) * rho[i] * e[i];
   }
};

void func()
{
  Func my_functor;
  Arcane::arcaneParallelFor(0,n,&my_functor,&Func::exec);
}
```

Cette syntaxe est un peu verbeuse. Si le compilateur supporte la norme
C++11, il est possible d'utiliser les lambda function pour simplifier l'écriture:

```cpp
void func()
{
  Arcane::arcaneParallelFor(0,n,[&](Integer begin,Integer size){
     for( Integer i=begin; i<(begin+size); ++i )
       p[i] = (gamma[i]-1.0) * rho[i] * e[i];
  });
}
```

Une spécialisation existe pour les groupes d'entités.
Pour paralléliser une énumération sur un groupe comme le code suivant:

```cpp
void func()
{
  ENUMERATE_CELL(icell,my_group){
    p[icell] = (gamma[icell]-1.0) * rho[icell] * e[icell];
  }
}
```

Il faut écrire comme cela:

```cpp
using namespace Arcane;
class Func
{
  public:
   void exec(CellVectorView view)
   {
     ENUMERATE_CELL(icell,view){
       p[icell] = (gamma[icell]-1.0) * rho[icell] * e[icell];
     }
   }
};

void func()
{
  Func my_functor;
  arcaneParallelForeach(my_group,&my_functor,&Func::exec);
}
```

De même, avec le support du C++11, on peut simplifier:

```cpp
using namespace Arcane;
void func()
{
  arcaneParallelForeach(my_group,[&](CellVectorView cells){
    ENUMERATE_CELL(icell,cells){
      p[icell] = (gamma[icell]-1.0) * rho[icell] * e[icell];
    }
  });
}
```

Pour les boucles Arcane::arcaneParallelFor() et Arcane::arcaneParallelForeach(), il est possible
de passer en argument une instance de ParallelLoopOptions pour
configurer la boucle parallèle. Par exemple, il est possible de
spécifier la taille de l'intervalle pour découper la boucle:

```cpp
void func()
{
  Arcane::ParallelLoopOptions options;
  // Exécute la boucle par parties d'environ 50 mailles.
  options.setGrainSize(50);
  Arcane::arcaneParallelForeach(my_group,options,[&](Arcane::CellVectorView cells){
    ENUMERATE_CELL(icell,cells){
      p[icell] = (gamma[icell]-1.0) * rho[icell] * e[icell];
    }
  });
}
```

## Utilisation explicite des tâches {#arcanedoc_parallel_concurrency_task}

La création d'un tâche se fait via la fabrique de tâche. Il faut spécifier
en argument un fonctor de la même manière que les boucles parallèles:

```cpp
class Func
{
  public:
   void exec(const TaskContext& ctx)
   {
     // Execute la tâche.
   }
};

void func()
{
  Func my_functor
  Arcane::ITask* master_task = Arcane::TaskFactory::createTask(&my_functor,&Func::exec);
}
```

Une fois la tâche créée, il est possible de la lancer et d'attendre sa
terminaison via la méthode ITask::launchAndWait(). Pour des raisons de
simplicité, la tâche n'est pas lancée tant que cette méthode n'a pas
été appelée.

Il est possible de créer des sous-tâches à partir d'une première tâche
via la méthode Arcane::TaskFactory::createChildTask().
L'utilisateur doit gérer le lancement et l'attente des sous-tâches.
Par exemple:

```cpp
using namespace Arcane;
ITask* master_task = TaskFactory::createTask(...);
UniqueArray<ITask*> sub_tasks;
sub_tasks.add(TaskFactory::createChildTask(master_task,&my_functor,&Func::exec);
sub_tasks.add(TaskFactory::createChildTask(master_task,&my_functor,&Func::exec);
master_task->launchAndWait(sub_tasks);
```

L'exemple complet suivant montre l'implémentation du calcul d'une suite
de Fibonacci via le mécanisme des tâches.

```cpp
using namespace Arcane;
class Fibonnaci
{
public:
 const long n;
 long* const sum;
 Fibonnaci( long n_, long* sum_ ) : n(n_), sum(sum_)
 {}
 void execute(const TaskContext& context)
 {
   if( n<10 ) {
     *sum = SerialFib(n);
   }
   else {
     long x, y;
     Fibonnaci a(n-1,&x);
     Fibonnaci b(n-2,&y);
     ITask* child_tasks[2];
     ITask* parent_task = context.task();
     child_tasks[0] = TaskFactory::createChildTask(parent_task,&a,&Test5Fibonnaci::execute);
     child_tasks[1] = TaskFactory::createChildTask(parent_task,&b,&Test5Fibonnaci::execute);
     parent_task->launchAndWait(ConstArrayView<ITask*>(2,child_tasks));

     // Effectue la somme
     *sum = x+y;
   }
 }
 static long SerialFib( long n )
 {
   if( n<2 )
     return n;
   else
     return SerialFib(n-1)+SerialFib(n-2);
 }
 static long ParallelFib( long n )
 {
   long sum;
   Test5Fibonnaci a(n,&sum);
   ITask* task = TaskFactory::createTask(&a,&Test5Fibonnaci::execute);
   task->launchAndWait();
   return sum;
 }
};
```


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_intro
</span>
<span class="next_section_button">
\ref arcanedoc_parallel_simd
</span>
</div>