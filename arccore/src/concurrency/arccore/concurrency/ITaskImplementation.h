// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITaskImplementation.h                                       (C) 2000-2025 */
/*                                                                           */
/* Interface de gestion des tâches.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ITASKIMPLEMENTATION_H
#define ARCCORE_BASE_ITASKIMPLEMENTATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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
class ARCCORE_CONCURRENCY_EXPORT ITaskImplementation
{
 public:

  virtual ~ITaskImplementation() = default;

 public:

  /*!
   * \internal.
   * Initialise l'implémentation avec au maximum \a nb_thread.
   * Si \a nb_thread vaut 0, l'implémentation peut choisir
   * le nombre de thread automatiquement.
   * Cette méthode est interne à Arcane et ne doit être appelée
   * que lors de l'initialisation de l'exécution.
   */
  virtual void initialize(Int32 nb_thread) = 0;
  /*!
   * \internal.
   * Termine l'utilisation de l'implémentation.
   * Cette méthode doit être appelée en fin de calcul uniquement.
   */
  virtual void terminate() = 0;
  /*!
   * \brief Créé une tâche racine.
   * L'implémentation doit recopier la valeur de \a f qui est soit
   * un TaskFunctor, soit un TaskFunctorWithContext.
   */
  virtual ITask* createRootTask(ITaskFunctor* f) = 0;

  //! Exécute le fonctor \a f en concurrence.
  virtual void executeParallelFor(Integer begin, Integer size, const ParallelLoopOptions& options, IRangeFunctor* f) = 0;

  //! Exécute le fonctor \a f en concurrence.
  virtual void executeParallelFor(Integer begin, Integer size, Integer block_size, IRangeFunctor* f) = 0;

  //! Exécute le fonctor \a f en concurrence.
  virtual void executeParallelFor(Integer begin, Integer size, IRangeFunctor* f) = 0;

  //! Exécute la boucle \a loop_info en concurrence.
  virtual void executeParallelFor(const ParallelFor1DLoopInfo& loop_info) = 0;

  //! Exécute une boucle 1D en concurrence
  virtual void executeParallelFor(const ComplexForLoopRanges<1>& loop_ranges,
                                  const ForLoopRunInfo& run_info,
                                  IMDRangeFunctor<1>* functor) = 0;
  //! Exécute une boucle 2D en concurrence
  virtual void executeParallelFor(const ComplexForLoopRanges<2>& loop_ranges,
                                  const ForLoopRunInfo& run_info,
                                  IMDRangeFunctor<2>* functor) = 0;
  //! Exécute une boucle 3D en concurrence
  virtual void executeParallelFor(const ComplexForLoopRanges<3>& loop_ranges,
                                  const ForLoopRunInfo& run_info,
                                  IMDRangeFunctor<3>* functor) = 0;
  //! Exécute une boucle 4D en concurrence
  virtual void executeParallelFor(const ComplexForLoopRanges<4>& loop_ranges,
                                  const ForLoopRunInfo& run_info,
                                  IMDRangeFunctor<4>* functor) = 0;

  //! Indique si l'implémentation est active.
  virtual bool isActive() const = 0;

  //! Nombre de threads utilisés au maximum pour gérer les tâches.
  virtual Int32 nbAllowedThread() const = 0;

  //! Implémentation de TaskFactory::currentTaskThreadIndex()
  virtual Int32 currentTaskThreadIndex() const = 0;

  //! Implémentation de TaskFactory::currentTaskIndex()
  virtual Int32 currentTaskIndex() const = 0;

  //! Affiche les informations sur le runtime utilisé
  virtual void printInfos(std::ostream& o) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
