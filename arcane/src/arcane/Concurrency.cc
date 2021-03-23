// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Concurrency.cc                                              (C) 2000-2016 */
/*                                                                           */
/* Classes gérant la concurrence (tâches, boucles parallèles, ...)           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceInfo.h"

#include "arcane/Observable.h"
#include "arcane/Concurrency.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SerialTask
: public ITask
{
 public:
  typedef TaskFunctor<SerialTask> TaskType;
 public:
  static const int FUNCTOR_CLASS_SIZE = sizeof(TaskType);
 public:
  SerialTask(ITaskFunctor* f)
  : m_functor(f)
  {
    // \a f doit être une instance de TaskFunctor<SerialTask>.
    // on recopie dans un buffer pré-dimensionné pour éviter
    // d'avoir à faire une allocation sur le tas via le new
    // classique. On utilise donc le new avec placement.

    TaskType* tf = (TaskType*)f;
    m_functor = new (functor_buf) TaskType(*tf);
  }
 public:
  virtual void launchAndWait()
  {
    if (m_functor){
      ITaskFunctor* tmp_f = m_functor;
      m_functor = 0;
      TaskContext task_context(this);
      tmp_f->executeFunctor(task_context);
      delete this;
    }
  }
  virtual void launchAndWait(ConstArrayView<ITask*> tasks)
  {
    for( Integer i=0,n=tasks.size(); i<n; ++i )
      tasks[i]->launchAndWait();
  }
  virtual ITask* _createChildTask(ITaskFunctor* functor)
  {
    return new SerialTask(functor);
  }
 private:
  ITaskFunctor* m_functor;
  char functor_buf[FUNCTOR_CLASS_SIZE];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NullTaskImplementation
: public ITaskImplementation
{
 public:
  static NullTaskImplementation singleton;
 public:
  void initialize(Int32 nb_thread) override
  {
    ARCANE_UNUSED(nb_thread);
  }
  void terminate() override
  {
  }
  ITask* createRootTask(ITaskFunctor* f) override
  {
    return new SerialTask(f);
  }
  void executeParallelFor(Integer begin,Integer size,Integer block_size,IRangeFunctor* f) override
  {
    ARCANE_UNUSED(block_size);
    f->executeFunctor(begin,size);
  }
  void executeParallelFor(Integer begin,Integer size,const ParallelLoopOptions& options,IRangeFunctor* f) override
  {
    ARCANE_UNUSED(options);
    f->executeFunctor(begin,size);
  }
  void executeParallelFor(Integer begin,Integer size,IRangeFunctor* f) override
  {
    f->executeFunctor(begin,size);
  }
  bool isActive() const override
  {
    return false;
  }
  Int32 nbAllowedThread() const override
  {
    return 1;
  }
  Int32 currentTaskThreadIndex() const override
  {
    return 0;
  }
  Int32 currentTaskIndex() const override
  {
    return 0;
  }
  void setDefaultParallelLoopOptions(const ParallelLoopOptions& v) override
  {
    m_default_loop_options = v;
  }

  const ParallelLoopOptions& defaultParallelLoopOptions() override
  {
    return m_default_loop_options;
  }

 private:
  ParallelLoopOptions m_default_loop_options;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NullTaskImplementation NullTaskImplementation::singleton;
ITaskImplementation* TaskFactory::m_impl = &NullTaskImplementation::singleton;
IObservable* TaskFactory::m_created_thread_observable = 0;
IObservable* TaskFactory::m_destroyed_thread_observable = 0;
Integer TaskFactory::m_verbose_level = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TaskFactory::
setImplementation(ITaskImplementation* task_impl)
{
  if (m_impl && m_impl!=&NullTaskImplementation::singleton)
    ARCANE_FATAL("TaskFactory already has an implementation");
  m_impl = task_impl;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IObservable*  TaskFactory::
createThreadObservable()
{
  if (!m_created_thread_observable)
    m_created_thread_observable = new Observable();
  return m_created_thread_observable;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IObservable*  TaskFactory::
destroyThreadObservable()
{
  if (!m_destroyed_thread_observable)
    m_destroyed_thread_observable = new Observable();
  return m_destroyed_thread_observable;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TaskFactory::
terminate()
{
  // C'est celui qui a positionné l'implémentation qui gère sa destruction.
  if (m_impl==&NullTaskImplementation::singleton)
    return;
  if (m_impl)
    m_impl->terminate();
  m_impl = &NullTaskImplementation::singleton;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \defgroup Concurrency Concurrence
 
 \brief Ensemble des classes gérant la concurrence.

 Pour plus de renseignements, se reporter à la page \ref arcanedoc_concurrency
*/

/*!
 * \page arcanedoc_concurrency Concurrence et multi-threading
 *
 * \tableofcontents

 La notion de concurrence est implémentée dans %Arcane via la notion de tâche.

 Cette notion de tâche permet l'exécution concurrente de plusieurs opérations via les threads.

 Cette notion est complémentaire de la notion de décomposition de domaine utilisée par le IParallelMng.
 Il est donc tout à fait possible de mélanger décomposition de domaine et les thread.

 \warning Néanmoins, si l'implémentation de IParallelMng se fait via MPI, il est déconseillé de faire des appels au IParallelMng
 lorsque des tâches se déroulent de manière concurrente, par exemle dans les boucles parallélisées. La plupart des
 implémentations MPI ne sont pas très performantes dans ce mode et certaines ne le supporte que partiellement.

 Pour utiliser les tâches, il faut inclure le fichier suivant:
 \code
 #include "arcane/Concurrency.h"
 \endcode

 Il existe deux mécanismes pour utiliser les tâches:
 <ol>
 <li>implicitement via la notion de boucle parallèle</li>
 <li>explicitement en créant les tâches directement</li>
 </ol>
 
 La première solution est la plus simple et doit être envisagée en priorité.

 \section arcanedoc_concurrency_activation Activation

 Par défaut, le support de la concurrence est désactivé. L'activation se fait <strong>avant</strong> le lancement du code, en spécifiant le
 nombre de tâches pouvant s'exécuter de manière concurrentes. Cela se fait en positionnant la variable d'environnement
 ARCANE_NB_TASK à la valeur souhaitée, par exemple 4.
 
 Il est possible de savoir dans le code si la concurrence est active en appelant la méthode TaskFactory::isActive().

 Il n'est pas possible d'activer la concurrence pendant l'exécution.

 \section arcanedoc_concurrency_parallel_for Boucles parallèles
 
 Il existe deux formes de boucles parallèles. La première forme s'applique
 sur les boucles classiques, la seconde sur les groupes d'entités.

 Le mécanisme de fonctionnement est similaire aux directives \c "omp parallel for" de OpenMp.

 \warning L'utilisateur de ce mécanisme doit s'assurer que la boucle peut être correctement parallélisée
 sans qu'il y ait d'effets de bord. Notamment, cela inclut (mais ne se limite pas) la garantie que les
 itérations de la boucle sont indépendantes, qu'il n'y a pas d'opérations de sortie de boucle (return, break).

 La première forme est pour paralléliser la boucle séquentielle suivante:

 \code
 * void func()
 * {
 *   for( Integer i=0; i<n; ++i )
 *     p[i] = (gamma[i]-1) * rho[i] * e[i];
 * }
 \endcode

 La parallélisation se fait comme suit: il faut d'abord écrire une classe fonctor qui
 représente l'opération que l'on souhaite effectuée sur un interval d'itération. Ensuite,
 il faut utiliser l'opération Parallel::For() en spécifiant ce fonctor en argument comme suit:

 \code
 * class Func
 * {
 *   public:
 *    void exec(Integer begin,Integer size)
 *    {
 *      for( Integer i=begin; i<(begin+size); ++i )
 *        p[i] = (gamma[i]-1) * rho[i] * e[i];
 *    }
 * };
 *
 * void func()
 * {
 *   Func my_functor;
 *   Parallel::For(0,n,&my_functor,&Func::exec);
 * }
 * \endcode

 Cette syntaxe est un peu verbeuse. Si le compilateur supporte la norme C++11, il est possible
 d'utiliser les lambda function pour simplifier l'écriture:
 
 \code
 * void func()
 * {
 *   Parallel::For(0,n,[&](Integer begin,Integer size){
 *      for( Integer i=begin; i<(begin+size); ++i )
 *        p[i] = (gamma[i]-1.0) * rho[i] * e[i];
 *   });
 * }
 \endcode

 Une spécialisation existe pour les groupes d'entités.
 Pour paralléliser une énumération sur un groupe comme le code suivant:

 \code
 * void func()
 * {
 *   ENUMERATE_CELL(icell,my_group){
 *     p[icell] = (gamma[icell]-1.0) * rho[icell] * e[icell];
 *   }
 * }
 \endcode

 Il faut écrire comme cela:

 \code
 * class Func
 * {
 *   public:
 *    void exec(CellVectorView view)
 *    {
 *      ENUMERATE_CELL(icell,view){
 *        p[icell] = (gamma[icell]-1.0) * rho[icell] * e[icell];
 *      }
 *    }
 * };
 *
 * void func()
 * {
 *   Func my_functor;
 *   Parallel::Foreach(my_group,&my_functor,&Func::exec);
 * }
 * \endcode

 De même, avec le support du C++11, on peut simplifier:

 \code
 * void func()
 * {
 *   Parallel::Foreach(my_group,[&](CellVectorView cells){
 *     ENUMERATE_CELL(icell,view){
 *       p[icell] = (gamma[icell]-1.0) * rho[icell] * e[icell];
 *     }
 *   });
 * }
 \endcode

 Pour les boucles Parallel::For et Parallel::Foreach, il est possible
 de passer en argument une instance de ParallelLoopOptions pour
 configurer la boucle parallèle. Par exemple, il est possible de
 spécifier la taille de l'intervalle pour découper la boucle:

 \code
 * void func()
 * {
 *   ParallelLoopOptions options;
 *   // Exécute la boucle par parties d'environ 50 mailles.
 *   options.setGrainSize(50);
 *   Parallel::Foreach(my_group,[&](CellVectorView cells){
 *     ENUMERATE_CELL(icell,view){
 *       p[icell] = (gamma[icell]-1.0) * rho[icell] * e[icell];
 *     }
 *   });
 * }
 \endcode


 \section arcanedoc_concurrency_task Utilisation explicite des tâches

 La création d'un tâche se fait via la fabrique de tâche. Il faut spécifier
 en argument un fonctor de la même manière que les boucles parallèles:

 \code
 * class Func
 * {
 *   public:
 *    void exec(const TaskContext& ctx)
 *    {
 *      // Execute la tâche.
 *    }
 * };
 *
 * void func()
 * {
 *   Func my_functor
 *   ITask* master_task = TaskFactory::createTask(&my_functor,&Func::exec);
 * }
 *
 \endcode
 
 Une fois la tâche créée, il est possible de la lancer et d'attendre
 sa terminaison via la méthode ITask::launchAndWait(). Pour des raisons de simplicité,
 la tâche n'est pas lancée tant que cette méthode n'a pas été appelée.

 Il est possible de créer des sous-tâches à partir d'une première tâche
 via la méthode TaskFactory::createChildTask().
 L'utilisateur doit gérer le lancement et l'attente des sous-tâches.
 Par exemple:

 \code
 ITask* master_task = TaskFactory::createTask(...);
 UniqueArray<ITask*> sub_tasks;
 sub_tasks.add(TaskFactory::createChildTask(master_task,&my_functor,&Func::exec);
 sub_tasks.add(TaskFactory::createChildTask(master_task,&my_functor,&Func::exec);
 master_task->launchAndWait(sub_tasks);
 \endcode

 L'exemple complet suivant montre l'implémentation du calcul d'une suite
 de Fibonacci via le mécanisme des tâches.

 \code
 * class Fibonnaci
 * {
 * public:
 *  const long n;
 *  long* const sum;
 *  Fibonnaci( long n_, long* sum_ ) : n(n_), sum(sum_)
 *  {}
 *  void execute(const TaskContext& context)
 *  {
 *    if( n<10 ) {
 *      *sum = SerialFib(n);
 *    }
 *    else {
 *      long x, y;
 *      Fibonnaci a(n-1,&x);
 *      Fibonnaci b(n-2,&y);
 *      ITask* child_tasks[2];
 *      ITask* parent_task = context.task();
 *      child_tasks[0] = TaskFactory::createChildTask(parent_task,&a,&Test5Fibonnaci::execute);
 *      child_tasks[1] = TaskFactory::createChildTask(parent_task,&b,&Test5Fibonnaci::execute);
 *      parent_task->launchAndWait(ConstArrayView<ITask*>(2,child_tasks));
 *
 *      // Effectue la somme
 *      *sum = x+y;
 *    }
 *  }
 *  static long SerialFib( long n )
 *  {
 *    if( n<2 )
 *      return n;
 *    else
 *      return SerialFib(n-1)+SerialFib(n-2);
 *  }
 *  static long ParallelFib( long n )
 *  {
 *    long sum;
 *    Test5Fibonnaci a(n,&sum);
 *    ITask* task = TaskFactory::createTask(&a,&Test5Fibonnaci::execute);
 *    task->launchAndWait();
 *    return sum;
 *  }
 * };
 \endcode


 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
