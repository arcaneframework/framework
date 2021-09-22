// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConcurrencyUtils.h                                          (C) 2000-2021 */
/*                                                                           */
/* Classes gérant la concurrence (tâches, boucles parallèles, ...)           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_CONCURRENCYUTILS_H
#define ARCANE_UTILS_CONCURRENCYUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/RangeFunctor.h"
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * TODO:
 * - Vérifier les fuites memoires
 * - BIEN INDIQUER QU'IL NE FAUT PLUS UTILISER UNE TACHE APRES LE WAIT!!!
 * - Regarder mecanisme pour les exceptions.
 * - Surcharger les For et Foreach sans specifier le block_size
 */

class ITaskImplementation;
class ITask;
class TaskFactory;
class IObservable;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Concurrency
 * \brief Options d'exécution d'une boucle parallèle en multi-thread.
 *
 * Cette classe permet de spécifier des paramètres d'exécution d'une
 * boucle parallèle.
 */
class ARCANE_UTILS_EXPORT ParallelLoopOptions
{
 private:
  //! Drapeau pour indiquer quels champs ont été positionnés.
  enum SetFlags
    {
      SF_MaxThread = 1,
      SF_GrainSize = 2,
      SF_Partitioner = 4
    };
 public:
  //! Type du partitionneur
  enum class Partitioner
    {
      //! Laisse le partitionneur géré le partitionnement et l'ordonnancement (défaut)
      Auto = 0,
      /*!
       * \brief Utilise un partitionnement statique.
       *
       * Dans ce mode, grainSize() n'est pas utilisé et le partitionnement ne
       * dépend que du nombre de threads et de l'intervalle d'itération.
       *
       * A noter que l'ordonnencement reste dynamique et donc du exécution à
       * l'autre ce n'est pas forcément le même thread qui va exécuter
       * le même bloc d'itération.
       */
      Static = 1,
      /*!
       * \brief Utilise un partitionnement et un ordonnancement statique.
       *
       * Ce mode est similaire à Partitioner::Static mais l'ordonnancement
       * est déterministe pour l'attribution des tâches: la valeur
       * renvoyée par TaskFactory::currentTaskIndex() est déterministe.
       */
      Deterministic = 2
    };
 public:

  ParallelLoopOptions()
  : m_grain_size(0),
    m_max_thread(-1), m_partitioner(Partitioner::Auto), m_flags(0){}

 public:

  //! Nombre maximal de threads autorisés.
  Integer maxThread() const { return m_max_thread; }
  /*!
   * \brief Positionne le nombre maximal de threads autorisé.
   *
   * Si \a v vaut 0 ou 1, l'exécution sera séquentielle.
   * Si \a v est supérieur à TaskFactory::nbAllowedThread(), c'est
   * cette dernière valeur qui sera utilisée.
   */
  void setMaxThread(Integer v)
  {
    m_max_thread = v;
    m_flags |= SF_MaxThread;
  }

  //! Taille d'un intervalle d'itération.
  Integer grainSize() const { return m_grain_size; }
  //! Positionne la taille (approximative) d'un intervalle d'itération
  void setGrainSize(Integer v)
  {
    m_grain_size = v;
    m_flags |= SF_GrainSize;
  }

  //! Type du partitionneur
  Partitioner partitioner() const { return m_partitioner; }
  //! Positionne le type du partitionneur
  void setPartitioner(Partitioner v)
  {
    m_partitioner = v;
    m_flags |= SF_Partitioner;
  }

 public:
  //! Fusionne les valeurs non modifiées de l'instance par celles de \a po.
  void mergeUnsetValues(const ParallelLoopOptions& po)
  {
    if (!(m_flags & SF_MaxThread))
      setMaxThread(po.maxThread());
    if (!(m_flags & SF_GrainSize))
      setGrainSize(po.grainSize());
    if (!(m_flags & SF_Partitioner))
      setPartitioner(po.partitioner());
 }
 private:

  Integer m_grain_size; //!< Taille d'un bloc de la boucle
  Integer m_max_thread; //!< Nombre maximum de threads pour la boucle
  Partitioner m_partitioner; //!< Type de partitionneur.
  int m_flags;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Contexte d'éxecution d'une tâche.
 * \ingroup Concurrency
 */
class ARCANE_UTILS_EXPORT TaskContext
{
 public:
  TaskContext(ITask* atask) : m_task(atask) {}
 public:
  //! Tâche courante.
  ITask* task() const { return m_task; }
 private:
  ITask* m_task;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un fonctor pour une tâche.
 * \ingroup Concurrency
 */
class ARCANE_UTILS_EXPORT ITaskFunctor
{
 public:
  virtual ~ITaskFunctor() = default;
 public:
  //! Exécute la méthode associé
  virtual void executeFunctor(const TaskContext& tc) =0;
  virtual ITaskFunctor* clone(void* buffer,Integer size) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fonctor sans argument pour une tâche.
 * \ingroup Concurrency
 */
template<typename InstanceType>
class TaskFunctor
: public ITaskFunctor
{
 public:
  typedef void (InstanceType::*FunctorType)();
 public:
  TaskFunctor(InstanceType* instance,FunctorType func)
  : m_instance(instance), m_function(func)
  {
  }
  TaskFunctor(const TaskFunctor& rhs) = default;
  TaskFunctor& operator=(const TaskFunctor& rhs) = delete;
 public:
  //! Exécute la méthode associé
  void executeFunctor(const TaskContext& /*tc*/) override
  {
    (m_instance->*m_function)();
  }
  ITaskFunctor* clone(void* buffer,Integer size) override
  {
    if (sizeof(*this)>(size_t)size)
      ARCANE_FATAL("INTERNAL: task functor buffer is too small");
    return new (buffer) TaskFunctor<InstanceType>(*this);
  }
 private:
  InstanceType* m_instance;
  FunctorType m_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fonctor pour une tâche prenant un TaskContext en argument.
 * \ingroup Concurrency
 */
template<typename InstanceType>
class TaskFunctorWithContext
: public ITaskFunctor
{
 public:
  typedef void (InstanceType::*FunctorType)(const TaskContext& tc);
 public:
  TaskFunctorWithContext(InstanceType* instance,FunctorType func)
  : m_instance(instance), m_function(func)
  {
  }
 public:
  //! Exécute la méthode associé
  void executeFunctor(const TaskContext& tc) override
  {
    (m_instance->*m_function)(tc);
  }
  ITaskFunctor* clone(void* buffer,Integer size) override
  {
    if (sizeof(*this)>(size_t)size)
      ARCANE_FATAL("INTERNAL: task functor buffer is too small");
    return new (buffer) TaskFunctorWithContext<InstanceType>(*this);
  }
 private:
  InstanceType* m_instance;
  FunctorType m_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Concurrency
 * \brief Interface d'une tâche concourante.
 *
 * Les tâches sont créées via TaskFactory.
 */
class ARCANE_UTILS_EXPORT ITask
{
  friend class TaskFactory;

 public:

  virtual ~ITask(){}

 public:
  /*!
   * \brief Lance la tâche et bloque jusqu'à ce qu'elle se termine.
   *
   * Après appel à cette fonction, la tâche est détruite et ne doit
   * plus être utilisée.
   */
  virtual void launchAndWait() =0;
  /*!
   * \brief Lance les tâches filles \a tasks et bloque
   * jusqu'à ce qu'elles se terminent.
   */
  virtual void launchAndWait(ConstArrayView<ITask*> tasks) =0;

 protected:

  virtual ITask* _createChildTask(ITaskFunctor* functor) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation d'une fabrique de tâches.
 *
 * \ingroup Concurrency
 *
 * Cette classe est interne à Arcane. Pour gérer les tâches, il
 * faut utiliser la classe TaskFactory.
 */
class ARCANE_UTILS_EXPORT ITaskImplementation
{
 public:
  virtual ~ITaskImplementation(){}
 public:
  /*!
   * \internal.
   * Initialise l'implémentation avec au maximum \a nb_thread.
   * Si \a nb_thread vaut 0, l'implémentation peut choisir
   * le nombre de thread automatiquement.
   * Cette méthode est interne à Arcane et ne doit être appelée
   * que lors de l'initialisation de l'exécution.
   */
  virtual void initialize(Int32 nb_thread) =0;
  /*!
   * \internal.
   * Termine l'utilisation de l'implémentation.
   * Cette méthode doit être appelée en fin de calcul uniquement.
   */
  virtual void terminate() =0;
  /*!
   * \brief Créé une tâche racine.
   * L'implémentation doit recopier la valeur de \a f qui est soit
   * un TaskFunctor, soit un TaskFunctorWithContext.
   */
  virtual ITask* createRootTask(ITaskFunctor* f) =0;

  //! Exécute le fonctor \a f en concurrence.
  virtual void executeParallelFor(Integer begin,Integer size,const ParallelLoopOptions& options,IRangeFunctor* f) =0;

  //! Exécute le fonctor \a f en concurrence.
  virtual void executeParallelFor(Integer begin,Integer size,Integer block_size,IRangeFunctor* f) =0;

  //! Exécute le fonctor \a f en concurrence.
  virtual void executeParallelFor(Integer begin,Integer size,IRangeFunctor* f) =0;

  //! Indique si l'implémentation est active.
  virtual bool isActive() const =0;

  //! Nombre de threads utilisés au maximum pour gérer les tâches.
  virtual Int32 nbAllowedThread() const =0;

  //! Implémentation de TaskFactory::currentTaskThreadIndex()
  virtual Int32 currentTaskThreadIndex() const =0;

  //! Implémentation de TaskFactory::currentTaskIndex()
  virtual Int32 currentTaskIndex() const =0;

  //! Positionne les valeurs par défaut d'exécution d'une boucle parallèle
  virtual void setDefaultParallelLoopOptions(const ParallelLoopOptions& v) =0;

  //! Valeurs par défaut d'exécution d'une boucle parallèle
  virtual const ParallelLoopOptions& defaultParallelLoopOptions() =0;

  virtual void printInfos(std::ostream& o) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fabrique pour les tâches.
 * \ingroup Concurrency
 */
class ARCANE_UTILS_EXPORT TaskFactory
{
 private:
  TaskFactory();
 public:
 public:

  /*!
   * \brief Créé une tâche.
   * Lors de l'exécution, la tâche appellera la méthode \a function via
   * l'instance \a instance.
   */
  template<typename InstanceType> static ITask*
  createTask(InstanceType* instance,void (InstanceType::*function)(const TaskContext& tc))
  {
    TaskFunctorWithContext<InstanceType> functor(instance,function);
    return m_impl->createRootTask(&functor);
  }

  /*!
   * \brief Créé une tâche.
   * Lors de l'exécution, la tâche appellera la méthode \a function via
   * l'instance \a instance.
   */
  template<typename InstanceType> static ITask*
  createTask(InstanceType* instance,void (InstanceType::*function)())
  {
    TaskFunctor<InstanceType> functor(instance,function);
    return m_impl->createRootTask(&functor);
  }

  /*!
   * \brief Créé une tâche fille.
   *
   * Lors de l'exécution, la tâche appellera la méthode \a function via
   * l'instance \a instance.
   */
  template<typename InstanceType> static ITask*
  createChildTask(ITask* parent_task,InstanceType* instance,void (InstanceType::*function)(const TaskContext& tc))
  {
    TaskFunctorWithContext<InstanceType> functor(instance,function);
    return parent_task->_createChildTask(&functor);
  }

  /*!
   * \brief Créé une tâche fille.
   *
   * Lors de l'exécution, la tâche appellera la méthode \a function via
   * l'instance \a instance.
   */
  template<typename InstanceType> static ITask*
  createChildTask(ITask* parent_task,InstanceType* instance,void (InstanceType::*function)())
  {
    TaskFunctor<InstanceType> functor(instance,function);
    return parent_task->_createChildTask(&functor);
  }

  //! Exécute le fonctor \a f en concurrence.
  static void executeParallelFor(Integer begin,Integer size,const ParallelLoopOptions& options,IRangeFunctor* f)
  {
    return m_impl->executeParallelFor(begin,size,options,f);
  }

  //! Exécute le fonctor \a f en concurrence.
  static void executeParallelFor(Integer begin,Integer size,Integer block_size,IRangeFunctor* f)
  {
    return m_impl->executeParallelFor(begin,size,block_size,f);
  }

  //! Exécute le fonctor \a f en concurrence.
  static void executeParallelFor(Integer begin,Integer size,IRangeFunctor* f)
  {
    return m_impl->executeParallelFor(begin,size,f);
  }

  //! Nombre de threads utilisés au maximum pour gérer les tâches.
  static Int32 nbAllowedThread()
  {
    return m_impl->nbAllowedThread();
  }

  /*!
   * \brief Indice (entre 0 et nbAllowedThread()-1) du thread exécutant la tâche actuelle.
   *
   * Pour des raisons de performance, il est préférable d'appeler cette méthode
   * le moins possible. L'idéal est de ne le faire qu'au début de l'exécution de la tâche
   * et ensuite d'utiliser la valeur retournée.
   */
  static Int32 currentTaskThreadIndex()
  {
    return m_impl->currentTaskThreadIndex();
  }

  /*!
   * \brief Indice (entre 0 et nbAllowedThread()-1) de la tâche actuelle.
   *
   * Cet indice est le même que currentTaskThreadIndex() sauf dans le cas
   * où on se trouve dans un executeParallelFor() avec un partitionnement
   * déterministe (ParallelLoopOptions::Partitioner::Deterministic).
   * Dans ce dernier cas, le numéro de la tâche est assigné de manière
   * déterministe qui ne dépend que du nombre de threads alloués pour la
   * tâche et de ParallelLoopOptions::grainSize().
   *
   * Si le thread courant n'exécute pas une tâche associé à cette implémentation,
   * retourne (-1).
   */
  static Int32 currentTaskIndex()
  {
    return m_impl->currentTaskIndex();
  }

 public:
  //! Positionne les valeurs par défaut d'exécution d'une boucle parallèle
  static void setDefaultParallelLoopOptions(const ParallelLoopOptions& v)
  {
    m_impl->setDefaultParallelLoopOptions(v);
  }
  //! Valeurs par défaut d'exécution d'une boucle parallèle
  static const ParallelLoopOptions& defaultParallelLoopOptions()
  {
    return m_impl->defaultParallelLoopOptions();
  }
 public:
  /*!
   * \brief Indique si les tâches sont actives.
   * Les tâches sont actives si une implémentation est disponible et si le nombre
   * de threads demandé est strictement supérieur à 1.
   */
  static bool isActive()
  {
    return m_impl->isActive();
  }

  /*!
   * \brief Affiche les informations sur l'implémentation.
   *
   * Les informations sont par exemple le numéro de version ou le nom
   * de l'implémentation.
   */
  static void printInfos(std::ostream& o)
  {
    return m_impl->printInfos(o);
  }

  /*!
   * \brief Observable appelé lors de la création d'un thread pour une tâche.
   *
   * \warning L'instance de l'observable est créée lors du premier appel
   * à cette méthode. Elle n'est donc pas thread-safe. De même,
   * la modification de l'observable (ajout/suppression d'observateur)
   * n'est pas thread-safe.
   */
  static IObservable* createThreadObservable();

  /*!
   * \brief Observable appelé lors de la destruction d'un thread pour une tâche.
   *
   * \warning L'instance de l'observable est créée lors du premier appel
   * à cette méthode. Elle n'est donc pas thread-safe. De même,
   * la modification de l'observable (ajout/suppression d'observateur)
   * n'est pas thread-safe.
   */
  static IObservable* destroyThreadObservable();

  /*!
   * \internal
   * \brief Indique qu'on n'utilisera plus les threads.
   * Cette méthode ne doit pas être appelée lorsque des tâches sont actives.
   */
  static void terminate();

 public:

  //! Positionne le niveau de verbosité (0 pour pas d'affichage, 1 par défaut)
  static void setVerboseLevel(Integer v) { m_verbose_level = v; }

  //! Niveau de verbosité
  static Integer verboseLevel() { return m_verbose_level; }

 public:
  //! \internal
  static void setImplementation(ITaskImplementation* task_impl);
 private:
  static ITaskImplementation* m_impl;
  static IObservable* m_created_thread_observable;
  static IObservable* m_destroyed_thread_observable;
  static Integer m_verbose_level;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
