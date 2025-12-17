// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TaskFactory.h                                               (C) 2000-2025 */
/*                                                                           */
/* Fabrique pour les tâches.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_TASKFACTORY_H
#define ARCCORE_CONCURRENCY_TASKFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ConcurrencyBase.h"
#include "arccore/concurrency/Task.h"
#include "arccore/concurrency/ITaskImplementation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Concurrency
 * \brief Fabrique pour les tâches.
 */
class ARCCORE_CONCURRENCY_EXPORT TaskFactory
{
  friend TaskFactoryInternal;

 public:

  TaskFactory() = delete;

 public:

  /*!
   * \brief Créé une tâche.
   * Lors de l'exécution, la tâche appellera la méthode \a function via
   * l'instance \a instance.
   */
  template <typename InstanceType> static ITask*
  createTask(InstanceType* instance, void (InstanceType::*function)(const TaskContext& tc))
  {
    TaskFunctorWithContext<InstanceType> functor(instance, function);
    return m_impl->createRootTask(&functor);
  }

  /*!
   * \brief Créé une tâche.
   * Lors de l'exécution, la tâche appellera la méthode \a function via
   * l'instance \a instance.
   */
  template <typename InstanceType> static ITask*
  createTask(InstanceType* instance, void (InstanceType::*function)())
  {
    TaskFunctor<InstanceType> functor(instance, function);
    return m_impl->createRootTask(&functor);
  }

  /*!
   * \brief Créé une tâche fille.
   *
   * Lors de l'exécution, la tâche appellera la méthode \a function via
   * l'instance \a instance.
   */
  template <typename InstanceType> static ITask*
  createChildTask(ITask* parent_task, InstanceType* instance, void (InstanceType::*function)(const TaskContext& tc))
  {
    ARCCORE_CHECK_POINTER(parent_task);
    TaskFunctorWithContext<InstanceType> functor(instance, function);
    return parent_task->_createChildTask(&functor);
  }

  /*!
   * \brief Créé une tâche fille.
   *
   * Lors de l'exécution, la tâche appellera la méthode \a function via
   * l'instance \a instance.
   */
  template <typename InstanceType> static ITask*
  createChildTask(ITask* parent_task, InstanceType* instance, void (InstanceType::*function)())
  {
    ARCCORE_CHECK_POINTER(parent_task);
    TaskFunctor<InstanceType> functor(instance, function);
    return parent_task->_createChildTask(&functor);
  }

  //! Exécute le fonctor \a f en concurrence.
  static void executeParallelFor(Integer begin, Integer size, const ParallelLoopOptions& options, IRangeFunctor* f)
  {
    m_impl->executeParallelFor(begin, size, options, f);
  }

  //! Exécute le fonctor \a f en concurrence.
  static void executeParallelFor(Integer begin, Integer size, Integer block_size, IRangeFunctor* f)
  {
    m_impl->executeParallelFor(begin, size, block_size, f);
  }

  //! Exécute le fonctor \a f en concurrence.
  static void executeParallelFor(Integer begin, Integer size, IRangeFunctor* f)
  {
    m_impl->executeParallelFor(begin, size, f);
  }

  //! Exécute la boucle \a loop_info en concurrence.
  static void executeParallelFor(const ParallelFor1DLoopInfo& loop_info)
  {
    m_impl->executeParallelFor(loop_info);
  }

  //! Exécute une boucle simple
  static void executeParallelFor(const ComplexForLoopRanges<1>& loop_ranges,
                                 const ParallelLoopOptions& options,
                                 IMDRangeFunctor<1>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, ForLoopRunInfo(options), functor);
  }

  //! Exécute une boucle simple
  static void executeParallelFor(const ComplexForLoopRanges<1>& loop_ranges,
                                 const ForLoopRunInfo& run_info,
                                 IMDRangeFunctor<1>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, run_info, functor);
  }

  //! Exécute une boucle 2D
  static void executeParallelFor(const ComplexForLoopRanges<2>& loop_ranges,
                                 const ParallelLoopOptions& options,
                                 IMDRangeFunctor<2>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, ForLoopRunInfo(options), functor);
  }

  //! Exécute une boucle 2D
  static void executeParallelFor(const ComplexForLoopRanges<2>& loop_ranges,
                                 const ForLoopRunInfo& run_info,
                                 IMDRangeFunctor<2>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, run_info, functor);
  }

  //! Exécute une boucle 3D
  static void executeParallelFor(const ComplexForLoopRanges<3>& loop_ranges,
                                 const ParallelLoopOptions& options,
                                 IMDRangeFunctor<3>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, ForLoopRunInfo(options), functor);
  }

  //! Exécute une boucle 3D
  static void executeParallelFor(const ComplexForLoopRanges<3>& loop_ranges,
                                 const ForLoopRunInfo& run_info,
                                 IMDRangeFunctor<3>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, run_info, functor);
  }

  //! Exécute une boucle 4D
  static void executeParallelFor(const ComplexForLoopRanges<4>& loop_ranges,
                                 const ParallelLoopOptions& options,
                                 IMDRangeFunctor<4>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, ForLoopRunInfo(options), functor);
  }

  //! Exécute une boucle 4D
  static void executeParallelFor(const ComplexForLoopRanges<4>& loop_ranges,
                                 const ForLoopRunInfo& run_info,
                                 IMDRangeFunctor<4>* functor)
  {
    m_impl->executeParallelFor(loop_ranges, run_info, functor);
  }

  //! Nombre de threads utilisés au maximum pour gérer les tâches.
  static Int32 nbAllowedThread() { return ConcurrencyBase::maxAllowedThread(); }

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

  // TODO: rendre ces deux méthodes obsolètes et indiquer d'utiliser
  // celles de ConcurrencyBase à la place.

  //! Positionne les valeurs par défaut d'exécution d'une boucle parallèle
  static void setDefaultParallelLoopOptions(const ParallelLoopOptions& v)
  {
    ConcurrencyBase::setDefaultParallelLoopOptions(v);
  }

  //! Valeurs par défaut d'exécution d'une boucle parallèle
  static const ParallelLoopOptions& defaultParallelLoopOptions()
  {
    return ConcurrencyBase::defaultParallelLoopOptions();
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
  ARCCORE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. Do not use it")
  static IObservable* createThreadObservable();

  /*!
   * \brief Observable appelé lors de la destruction d'un thread pour une tâche.
   *
   * \warning L'instance de l'observable est créée lors du premier appel
   * à cette méthode. Elle n'est donc pas thread-safe. De même,
   * la modification de l'observable (ajout/suppression d'observateur)
   * n'est pas thread-safe.
   */
  ARCCORE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. Do not use it")
  static IObservable* destroyThreadObservable();

  /*!
   * \internal
   * \brief Indique qu'on n'utilisera plus les threads.
   * Cette méthode ne doit pas être appelée lorsque des tâches sont actives.
   */
  static void terminate();

 public:

  //! Positionne le niveau de verbosité (0 pour pas d'affichage qui est le défaut)
  static void setVerboseLevel(Integer v) { m_verbose_level = v; }

  //! Niveau de verbosité
  static Integer verboseLevel() { return m_verbose_level; }

 public:

  //! \internal
  ARCCORE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. "
                            "Use TaskFactoryInternal::setImplementation() instead")
  static void _internalSetImplementation(ITaskImplementation* task_impl);

 private:

  static ITaskImplementation* m_impl;
  static Int32 m_verbose_level;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
