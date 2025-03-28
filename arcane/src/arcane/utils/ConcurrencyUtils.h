// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConcurrencyUtils.h                                          (C) 2000-2025 */
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
#include "arcane/utils/ForLoopTraceInfo.h"
#include "arcane/utils/ParallelLoopOptions.h"

#include <optional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TaskFactoryInternal;

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
 * \brief Informations d'exécution d'une boucle.
 *
 * Cette classe permet de gérer les informations d'exécutions communes à toutes
 * les boucles.
 */
class ARCANE_UTILS_EXPORT ForLoopRunInfo
{
 public:

  using ThatClass = ForLoopRunInfo;

 public:

  ForLoopRunInfo() = default;
  explicit ForLoopRunInfo(const ParallelLoopOptions& options)
  : m_options(options) {}
  ForLoopRunInfo(const ParallelLoopOptions& options,const ForLoopTraceInfo& trace_info)
  : m_options(options), m_trace_info(trace_info) {}
  explicit ForLoopRunInfo(const ForLoopTraceInfo& trace_info)
  : m_trace_info(trace_info) {}

 public:

  std::optional<ParallelLoopOptions> options() const { return m_options; }
  ThatClass& addOptions(const ParallelLoopOptions& v) { m_options = v; return (*this); }
  const ForLoopTraceInfo& traceInfo() const { return m_trace_info; }
  ThatClass& addTraceInfo(const ForLoopTraceInfo& v) { m_trace_info = v; return (*this); }

  /*!
   * \brief Positionne le pointeur conservant les statistiques d'exécution.
   *
   * Ce pointeur \a v doit rester valide durant toute l'exécution de la boucle.
   */
  void setExecStat(ForLoopOneExecStat* v) { m_exec_stat = v; }

  //! Pointeur contenant les statistiques d'exécution.
  ForLoopOneExecStat* execStat() const { return m_exec_stat; }

 protected:

  std::optional<ParallelLoopOptions> m_options;
  ForLoopTraceInfo m_trace_info;
  ForLoopOneExecStat* m_exec_stat = nullptr;
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
  explicit TaskContext(ITask* atask) : m_task(atask) {}
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
 protected:
  ITaskFunctor(const ITaskFunctor&) = default;
  ITaskFunctor() = default;
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
  : ITaskFunctor(), m_instance(instance), m_function(func)
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

  //! Exécute la boucle \a loop_info en concurrence.
  virtual void executeParallelFor(const ParallelFor1DLoopInfo& loop_info) =0;

  //! Exécute une boucle 1D en concurrence
  virtual void executeParallelFor(const ComplexForLoopRanges<1>& loop_ranges,
                                  const ForLoopRunInfo& run_info,
                                  IMDRangeFunctor<1>* functor) =0;
  //! Exécute une boucle 2D en concurrence
  virtual void executeParallelFor(const ComplexForLoopRanges<2>& loop_ranges,
                                  const ForLoopRunInfo& run_info,
                                  IMDRangeFunctor<2>* functor) =0;
  //! Exécute une boucle 3D en concurrence
  virtual void executeParallelFor(const ComplexForLoopRanges<3>& loop_ranges,
                                  const ForLoopRunInfo& run_info,
                                  IMDRangeFunctor<3>* functor) =0;
  //! Exécute une boucle 4D en concurrence
  virtual void executeParallelFor(const ComplexForLoopRanges<4>& loop_ranges,
                                  const ForLoopRunInfo& run_info,
                                  IMDRangeFunctor<4>* functor) =0;

  //! Indique si l'implémentation est active.
  virtual bool isActive() const =0;

  //! Nombre de threads utilisés au maximum pour gérer les tâches.
  virtual Int32 nbAllowedThread() const =0;

  //! Implémentation de TaskFactory::currentTaskThreadIndex()
  virtual Int32 currentTaskThreadIndex() const =0;

  //! Implémentation de TaskFactory::currentTaskIndex()
  virtual Int32 currentTaskIndex() const =0;

  //! Affiche les informations sur le runtime utilisé
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
  friend TaskFactoryInternal;

 public:

  TaskFactory() = delete;

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
    ARCANE_CHECK_POINTER(parent_task);
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
    ARCANE_CHECK_POINTER(parent_task);
    TaskFunctor<InstanceType> functor(instance,function);
    return parent_task->_createChildTask(&functor);
  }

  //! Exécute le fonctor \a f en concurrence.
  static void executeParallelFor(Integer begin,Integer size,const ParallelLoopOptions& options,IRangeFunctor* f)
  {
    m_impl->executeParallelFor(begin,size,options,f);
  }

  //! Exécute le fonctor \a f en concurrence.
  static void executeParallelFor(Integer begin,Integer size,Integer block_size,IRangeFunctor* f)
  {
    m_impl->executeParallelFor(begin,size,block_size,f);
  }

  //! Exécute le fonctor \a f en concurrence.
  static void executeParallelFor(Integer begin,Integer size,IRangeFunctor* f)
  {
    m_impl->executeParallelFor(begin,size,f);
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
    m_impl->executeParallelFor(loop_ranges,ForLoopRunInfo(options),functor);
  }

  //! Exécute une boucle simple
  static void executeParallelFor(const ComplexForLoopRanges<1>& loop_ranges,
                                  const ForLoopRunInfo& run_info,
                                  IMDRangeFunctor<1>* functor)
  {
    m_impl->executeParallelFor(loop_ranges,run_info,functor);
  }

  //! Exécute une boucle 2D
  static void executeParallelFor(const ComplexForLoopRanges<2>& loop_ranges,
                                 const ParallelLoopOptions& options,
                                 IMDRangeFunctor<2>* functor)
  {
    m_impl->executeParallelFor(loop_ranges,ForLoopRunInfo(options),functor);
  }

  //! Exécute une boucle 2D
  static void executeParallelFor(const ComplexForLoopRanges<2>& loop_ranges,
                                 const ForLoopRunInfo& run_info,
                                 IMDRangeFunctor<2>* functor)
  {
    m_impl->executeParallelFor(loop_ranges,run_info,functor);
  }

  //! Exécute une boucle 3D
  static void executeParallelFor(const ComplexForLoopRanges<3>& loop_ranges,
                                 const ParallelLoopOptions& options,
                                 IMDRangeFunctor<3>* functor)
  {
    m_impl->executeParallelFor(loop_ranges,ForLoopRunInfo(options),functor);
  }

  //! Exécute une boucle 3D
  static void executeParallelFor(const ComplexForLoopRanges<3>& loop_ranges,
                                 const ForLoopRunInfo& run_info,
                                 IMDRangeFunctor<3>* functor)
  {
    m_impl->executeParallelFor(loop_ranges,run_info,functor);
  }

  //! Exécute une boucle 4D
  static void executeParallelFor(const ComplexForLoopRanges<4>& loop_ranges,
                                 const ParallelLoopOptions& options,
                                 IMDRangeFunctor<4>* functor)
  {
    m_impl->executeParallelFor(loop_ranges,ForLoopRunInfo(options),functor);
  }

  //! Exécute une boucle 4D
  static void executeParallelFor(const ComplexForLoopRanges<4>& loop_ranges,
                                 const ForLoopRunInfo& run_info,
                                 IMDRangeFunctor<4>* functor)
  {
    m_impl->executeParallelFor(loop_ranges,run_info,functor);
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
    m_default_loop_options = v;
  }

  //! Valeurs par défaut d'exécution d'une boucle parallèle
  static const ParallelLoopOptions& defaultParallelLoopOptions()
  {
    return m_default_loop_options;
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
  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. Do not use it")
  static IObservable* createThreadObservable();

  /*!
   * \brief Observable appelé lors de la destruction d'un thread pour une tâche.
   *
   * \warning L'instance de l'observable est créée lors du premier appel
   * à cette méthode. Elle n'est donc pas thread-safe. De même,
   * la modification de l'observable (ajout/suppression d'observateur)
   * n'est pas thread-safe.
   */
  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. Do not use it")
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
  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. "
                           "Use TaskFactoryInternal::setImplementation() instead")
  static void _internalSetImplementation(ITaskImplementation* task_impl);

 private:

  static ITaskImplementation* m_impl;
  static Int32 m_verbose_level;
  static ParallelLoopOptions m_default_loop_options;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Caractéristiques d'un boucle 1D multi-thread.
 *
 * Cette classe permet de spécifier les options d'une boucle à paralléliser
 * en mode multi-thread.
 */
class ARCANE_UTILS_EXPORT ParallelFor1DLoopInfo
{
 public:

 using ThatClass = ParallelFor1DLoopInfo;

 public:

  ParallelFor1DLoopInfo(Int32 begin,Int32 size,IRangeFunctor* functor)
  : m_begin(begin), m_size(size), m_functor(functor) {}
  ParallelFor1DLoopInfo(Int32 begin,Int32 size,IRangeFunctor* functor,const ForLoopRunInfo& run_info)
  : m_run_info(run_info), m_begin(begin), m_size(size), m_functor(functor) {}
  ParallelFor1DLoopInfo(Int32 begin,Int32 size, Int32 block_size,IRangeFunctor* functor)
  : m_begin(begin), m_size(size), m_functor(functor)
  {
    ParallelLoopOptions opts(TaskFactory::defaultParallelLoopOptions());
    opts.setGrainSize(block_size);
    m_run_info.addOptions(opts);
  }

 public:

  Int32 beginIndex() const { return m_begin; }
  Int32 size() const { return m_size; }
  IRangeFunctor* functor() const { return m_functor; }
  ForLoopRunInfo& runInfo() { return m_run_info; }
  const ForLoopRunInfo& runInfo() const { return m_run_info; }

 private:

  ForLoopRunInfo m_run_info;
  Int32 m_begin = 0;
  Int32 m_size = 0;
  IRangeFunctor* m_functor = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération donné par \a loop_ranges.
 */
template<int RankValue,typename LambdaType,typename... ReducerArgs> inline void
arcaneParallelFor(const ComplexForLoopRanges<RankValue>& loop_ranges,
                  const ForLoopRunInfo& run_info,
                  const LambdaType& lambda_function,
                  const ReducerArgs&... reducer_args)
{
  // Modif Arcane 3.7.9 (septembre 2022)
  // Effectue une copie pour privatiser au thread courant les valeurs de la lambda.
  // Cela est nécessaire pour que objets comme les reducers soient bien pris
  // en compte.
  // TODO: regarder si on pourrait faire la copie uniquement une fois par thread
  // si cette copie devient couteuse.
  // NOTE: A partir de la version 3.12.15 (avril 2024), avec la nouvelle version
  // des réducteurs (Reduce2), cette privatisation n'est plus utile. Une fois
  // qu'on aura supprimer les anciennes classes gérant les réductions (Reduce),
  // on pourra supprimer cette privatisation
  auto xfunc = [&lambda_function,reducer_args...] (const ComplexForLoopRanges<RankValue>& sub_bounds)
  {
    using Type = typename std::remove_reference<LambdaType>::type;
    Type private_lambda(lambda_function);
    arcaneSequentialFor(sub_bounds,private_lambda,reducer_args...);
  };
  LambdaMDRangeFunctor<RankValue,decltype(xfunc)> ipf(xfunc);
  TaskFactory::executeParallelFor(loop_ranges,run_info,&ipf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération donné par \a loop_ranges.
 */
template<int RankValue,typename LambdaType,typename... ReducerArgs> inline void
arcaneParallelFor(const ComplexForLoopRanges<RankValue>& loop_ranges,
                  const ParallelLoopOptions& options,
                  const LambdaType& lambda_function,
                  const ReducerArgs&... reducer_args)
{
  arcaneParallelFor(loop_ranges,ForLoopRunInfo(options),lambda_function,reducer_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération donné par \a loop_ranges.
 */
template <int RankValue, typename LambdaType, typename... ReducerArgs> inline void
arcaneParallelFor(const SimpleForLoopRanges<RankValue>& loop_ranges,
                  const ForLoopRunInfo& run_info,
                  const LambdaType& lambda_function,
                  const ReducerArgs&... reducer_args)
{
  ComplexForLoopRanges<RankValue> complex_loop_ranges{ loop_ranges };
  arcaneParallelFor(complex_loop_ranges, run_info, lambda_function, reducer_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération donné par \a loop_ranges.
 */
template <int RankValue, typename LambdaType, typename... ReducerArgs> inline void
arcaneParallelFor(const SimpleForLoopRanges<RankValue>& loop_ranges,
                  const ParallelLoopOptions& options,
                  const LambdaType& lambda_function,
                  const ReducerArgs&... reducer_args)
{
  ComplexForLoopRanges<RankValue> complex_loop_ranges{ loop_ranges };
  arcaneParallelFor(complex_loop_ranges, ForLoopRunInfo(options), lambda_function, reducer_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération donné par \a loop_ranges.
 */
template<int RankValue,typename LambdaType> inline void
arcaneParallelFor(const ComplexForLoopRanges<RankValue>& loop_ranges,
                  const LambdaType& lambda_function)
{
  ParallelLoopOptions options;
  arcaneParallelFor(loop_ranges,options,lambda_function);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique en concurrence la fonction lambda \a lambda_function
 * sur l'intervalle d'itération donné par \a loop_ranges.
 */
template<int RankValue,typename LambdaType> inline void
arcaneParallelFor(const SimpleForLoopRanges<RankValue>& loop_ranges,
                  const LambdaType& lambda_function)
{
  ParallelLoopOptions options;
  ComplexForLoopRanges<RankValue> complex_loop_ranges{loop_ranges};
  arcaneParallelFor(complex_loop_ranges,options,lambda_function);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
