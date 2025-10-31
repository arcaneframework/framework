// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Task.h                                                      (C) 2000-2025 */
/*                                                                           */
/* Classes gérant les tâches concurrentes.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_TASK_H
#define ARCCORE_BASE_TASK_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/RangeFunctor.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/ForLoopTraceInfo.h"
#include "arccore/base/ParallelLoopOptions.h"
#include "arccore/base/ForLoopRunInfo.h"

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Contexte d'éxecution d'une tâche.
 * \ingroup Concurrency
 */
class ARCCORE_CONCURRENCY_EXPORT TaskContext
{
 public:

  explicit TaskContext(ITask* atask)
  : m_task(atask)
  {}

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
class ARCCORE_CONCURRENCY_EXPORT ITaskFunctor
{
 public:

  virtual ~ITaskFunctor() = default;

 protected:

  ITaskFunctor(const ITaskFunctor&) = default;
  ITaskFunctor() = default;

 public:

  //! Exécute la méthode associé
  virtual void executeFunctor(const TaskContext& tc) = 0;
  virtual ITaskFunctor* clone(void* buffer, Integer size) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fonctor sans argument pour une tâche.
 * \ingroup Concurrency
 */
template <typename InstanceType>
class TaskFunctor
: public ITaskFunctor
{
 public:

  typedef void (InstanceType::*FunctorType)();

 public:

  TaskFunctor(InstanceType* instance, FunctorType func)
  : m_instance(instance)
  , m_function(func)
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
  ITaskFunctor* clone(void* buffer, Integer size) override
  {
    if (sizeof(*this) > (size_t)size)
      ARCCORE_FATAL("INTERNAL: task functor buffer is too small");
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
template <typename InstanceType>
class TaskFunctorWithContext
: public ITaskFunctor
{
 public:

  typedef void (InstanceType::*FunctorType)(const TaskContext& tc);

 public:

  TaskFunctorWithContext(InstanceType* instance, FunctorType func)
  : ITaskFunctor()
  , m_instance(instance)
  , m_function(func)
  {
  }

 public:

  //! Exécute la méthode associé
  void executeFunctor(const TaskContext& tc) override
  {
    (m_instance->*m_function)(tc);
  }
  ITaskFunctor* clone(void* buffer, Integer size) override
  {
    if (sizeof(*this) > (size_t)size)
      ARCCORE_FATAL("INTERNAL: task functor buffer is too small");
    return new (buffer) TaskFunctorWithContext<InstanceType>(*this);
  }

 private:

  InstanceType* m_instance = nullptr;
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
class ARCCORE_CONCURRENCY_EXPORT ITask
{
  friend class TaskFactory;

 public:

  virtual ~ITask() = default;

 public:

  /*!
   * \brief Lance la tâche et bloque jusqu'à ce qu'elle se termine.
   *
   * Après appel à cette fonction, la tâche est détruite et ne doit
   * plus être utilisée.
   */
  virtual void launchAndWait() = 0;
  /*!
   * \brief Lance les tâches filles \a tasks et bloque
   * jusqu'à ce qu'elles se terminent.
   */
  virtual void launchAndWait(ConstArrayView<ITask*> tasks) = 0;

 protected:

  virtual ITask* _createChildTask(ITaskFunctor* functor) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
