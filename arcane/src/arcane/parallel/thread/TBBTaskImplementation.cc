// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TBBTaskImplementation.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Implémentation des tâches utilisant TBB (Intel Threads Building Blocks).  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IThreadImplementation.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/IFunctor.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/ForLoopRanges.h"
#include "arcane/utils/ConcurrencyUtils.h"
#include "arcane/utils/IObservable.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Profiling.h"
#include "arcane/utils/MemoryAllocator.h"
#include "arcane/utils/FixedArray.h"
#include "arccore/concurrency/internal/TaskFactoryInternal.h"
#include "arcane/utils/internal/DependencyInjection.h"

#include "arcane/core/FactoryService.h"

#include <new>
#include <stack>

// Il faut définir cette macro pour que la classe 'blocked_rangeNd' soit disponible

#define TBB_PREVIEW_BLOCKED_RANGE_ND 1

// la macro 'ARCANE_USE_ONETBB' est définie dans le CMakeLists.txt
// si on compile avec la version OneTBB version 2021+
// (https://github.com/oneapi-src/oneTBB.git)
// A terme ce sera la seule version supportée par Arcane.

#ifdef ARCANE_USE_ONETBB

// Nécessaire pour avoir accès à task_scheduler_handle
#define TBB_PREVIEW_WAITING_FOR_WORKERS 1
#include <tbb/tbb.h>
#include <oneapi/tbb/concurrent_set.h>
#include <oneapi/tbb/global_control.h>

#else // ARCANE_USE_ONETBB

// NOTE GG: depuis mars 2019, la version 2018.3+ des TBB est obligatoire.
// C'est celle qui introduit 'blocked_rangeNd.h'
#include <tbb/tbb.h>
#if __has_include(<tbb/blocked_rangeNd.h>)
#include <tbb/blocked_rangeNd.h>
#endif

/*
 * Maintenant vérifie que la version est au moins 2018.3.
 */

// Pour TBB 2018, la valeur de TBB_VERSION_MINOR vaut toujours 0 même
// pour TBB 2018 Update 3. On ne peut donc pas utiliser TBB_VERSION_MINO
// pour savoir exactement quelle version de TBB 2018 on utilise.
// A la place on utilise TBB_INTERFACE_VERSION qui est incrémenté
// quand l'interface change. Dans notre cas pour la version 2018.3 elle
// vaut 10003. On indique donc que la version de TBB est trop ancienne
// si TBB_INTERFACE_VERSION est plus petit que 10003
#if (TBB_INTERFACE_VERSION < 10003)
#define ARCANE_OLD_TBB
#endif

#ifdef ARCANE_OLD_TBB
#  if defined(__GNUG__)
#    define ARCANE_STR_HELPER(x) #x
#    define ARCANE_STR(x) ARCANE_STR_HELPER(x)
#    pragma message "Your version of TBB is : " ARCANE_STR(TBB_VERSION_MAJOR) "." ARCANE_STR(TBB_VERSION_MINOR)
#  endif
#  error "Your version of TBB is too old. TBB 2018.3+ is required. Please disable TBB in configuration using -DCMAKE_DISABLE_FIND_PACKAGE_TBB=TRUE"
#endif

#include <thread>
#include <mutex>

#endif // ARCANE_USE_ONETBB

namespace
{
#if (TBB_VERSION_MAJOR > 2022) || (TBB_VERSION_MAJOR == 2022 && TBB_VERSION_MINOR > 0) || defined __TBB_blocked_nd_range_H

// La classe "blocked_rangeNd" a été retirée dans la version
// 2022.0.0 et remplacée par "blocked_nd_range".
template <typename Value, unsigned int N>
using blocked_nd_range = tbb::blocked_nd_range<Value, N>;

#else

template <typename Value, unsigned int N>
using blocked_nd_range = tbb::blocked_rangeNd<Value, N>;

#endif
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class TBBTaskImplementation;

// TODO: utiliser un pool mémoire spécifique pour gérer les
// OneTBBTask pour optimiser les new/delete des instances de cette classe.
// Auparavant avec les anciennes versions de TBB cela était géré avec
// la méthode 'tbb::task::allocate_child()'.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

// Positif si on récupère les statistiques d'exécution
bool isStatActive()
{
  return ProfilingRegistry::hasProfiling();
}

/*!
 * \brief Classe permettant de garantir qu'on enregistre les statistiques
 * d'exécution même en cas d'exception.
 */
class ScopedExecInfo
{
 public:

  explicit ScopedExecInfo(const ForLoopRunInfo& run_info)
  : m_run_info(run_info)
  {
    // Si run_info.execInfo() n'est pas nul, on l'utilise.
    // Cela signifie que c'est l'appelant de qui va gérer les statistiques
    // d'exécution. Sinon, on utilise \a m_stat_info si les statistiques
    // d'exécution sont demandées.
    ForLoopOneExecStat* ptr = run_info.execStat();
    if (ptr){
      m_stat_info_ptr = ptr;
      m_use_own_run_info = false;
    }
    else
      m_stat_info_ptr = isStatActive() ? &m_stat_info : nullptr;
  }
  ~ScopedExecInfo()
  {
#ifdef PRINT_STAT_INFO
    if (m_stat_info_ptr){
      bool is_valid = m_run_info.traceInfo().isValid();
      if (!is_valid)
        std::cout << "ADD_OWN_RUN_INFO nb_chunk=" << m_stat_info_ptr->nbChunk()
                  << " stack=" << platform::getStackTrace()
                  << "\n";
      else
        std::cout << "ADD_OWN_RUN_INFO nb_chunk=" << m_stat_info_ptr->nbChunk()
                  << " trace_name=" << m_run_info.traceInfo().traceInfo().name() << "\n";
    }
#endif
    if (m_stat_info_ptr && m_use_own_run_info){
      ProfilingRegistry::_threadLocalForLoopInstance()->merge(*m_stat_info_ptr,m_run_info.traceInfo());
    }
  }

 public:

  ForLoopOneExecStat* statInfo() const { return m_stat_info_ptr; }
  bool isOwn() const { return m_use_own_run_info; }
 private:

  ForLoopOneExecStat m_stat_info;
  ForLoopOneExecStat* m_stat_info_ptr = nullptr;
  ForLoopRunInfo m_run_info;
  //! Indique si on utilise m_stat_info
  bool m_use_own_run_info = true;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline int _currentTaskTreadIndex()
{
  // NOTE: Avec OneTBB 2021, la valeur n'est plus '0' si on appelle cette méthode
  // depuis un thread en dehors d'un task_arena. Avec la version 2021,
  // la valeur est 65535.
  // NOTE: Il semble que cela soit un bug de la 2021.3.
  return tbb::this_task_arena::current_thread_index();
}

inline blocked_nd_range<Int32, 1>
_toTBBRange(const ComplexForLoopRanges<1>& r)
{
  return {{r.lowerBound<0>(), r.upperBound<0>()}};
}

inline blocked_nd_range<Int32, 2>
_toTBBRange(const ComplexForLoopRanges<2>& r)
{
  return {{r.lowerBound<0>(), r.upperBound<0>()},
          {r.lowerBound<1>(), r.upperBound<1>()}};

}

inline blocked_nd_range<Int32, 3>
_toTBBRange(const ComplexForLoopRanges<3>& r)
{
  return {{r.lowerBound<0>(), r.upperBound<0>()},
          {r.lowerBound<1>(), r.upperBound<1>()},
          {r.lowerBound<2>(), r.upperBound<2>()}};
}

inline blocked_nd_range<Int32, 4>
_toTBBRange(const ComplexForLoopRanges<4>& r)
{
  return {{r.lowerBound<0>(), r.upperBound<0>()},
          {r.lowerBound<1>(), r.upperBound<1>()},
          {r.lowerBound<2>(), r.upperBound<2>()},
          {r.lowerBound<3>(), r.upperBound<3>()}};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline blocked_nd_range<Int32, 2>
_toTBBRangeWithGrain(const blocked_nd_range<Int32, 2>& r, FixedArray<size_t, 2> grain_sizes)
{
  return {{r.dim(0).begin(), r.dim(0).end(), grain_sizes[0]},
          {r.dim(1).begin(), r.dim(1).end(), grain_sizes[1]}};
}

inline blocked_nd_range<Int32, 3>
_toTBBRangeWithGrain(const blocked_nd_range<Int32, 3>& r, FixedArray<size_t, 3> grain_sizes)
{
  return {{r.dim(0).begin(), r.dim(0).end(), grain_sizes[0]},
          {r.dim(1).begin(), r.dim(1).end(), grain_sizes[1]},
          {r.dim(2).begin(), r.dim(2).end(), grain_sizes[2]}};
}

inline blocked_nd_range<Int32, 4>
_toTBBRangeWithGrain(const blocked_nd_range<Int32, 4>& r, FixedArray<size_t, 4> grain_sizes)
{
  return {{r.dim(0).begin(), r.dim(0).end(), grain_sizes[0]},
          {r.dim(1).begin(), r.dim(1).end(), grain_sizes[1]},
          {r.dim(2).begin(), r.dim(2).end(), grain_sizes[2]},
          {r.dim(3).begin(), r.dim(3).end(), grain_sizes[3]}};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ComplexForLoopRanges<2>
_fromTBBRange(const blocked_nd_range<Int32, 2>& r)
{
  using BoundsType = ArrayBounds<MDDim2>;
  using ArrayExtentType = typename BoundsType::ArrayExtentType;

  BoundsType lower_bounds(ArrayExtentType(r.dim(0).begin(),r.dim(1).begin()));
  auto s0 = static_cast<Int32>(r.dim(0).size());
  auto s1 = static_cast<Int32>(r.dim(1).size());
  BoundsType sizes(ArrayExtentType(s0,s1));
  return { lower_bounds, sizes };
}

inline ComplexForLoopRanges<3>
_fromTBBRange(const blocked_nd_range<Int32, 3>& r)
{
  using BoundsType = ArrayBounds<MDDim3>;
  using ArrayExtentType = typename BoundsType::ArrayExtentType;

  BoundsType lower_bounds(ArrayExtentType(r.dim(0).begin(),r.dim(1).begin(),r.dim(2).begin()));
  auto s0 = static_cast<Int32>(r.dim(0).size());
  auto s1 = static_cast<Int32>(r.dim(1).size());
  auto s2 = static_cast<Int32>(r.dim(2).size());
  BoundsType sizes(ArrayExtentType(s0,s1,s2));
  return { lower_bounds, sizes };
}

inline ComplexForLoopRanges<4>
_fromTBBRange(const blocked_nd_range<Int32, 4>& r)
{
  using BoundsType = ArrayBounds<MDDim4>;
  using ArrayExtentType = typename BoundsType::ArrayExtentType;

  BoundsType lower_bounds(ArrayExtentType(r.dim(0).begin(),r.dim(1).begin(),r.dim(2).begin(),r.dim(3).begin()));
  auto s0 = static_cast<Int32>(r.dim(0).size());
  auto s1 = static_cast<Int32>(r.dim(1).size());
  auto s2 = static_cast<Int32>(r.dim(2).size());
  auto s3 = static_cast<Int32>(r.dim(3).size());
  BoundsType sizes(ArrayExtentType(s0,s1,s2,s3));
  return { lower_bounds, sizes };
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_USE_ONETBB

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OneTBBTaskFunctor
{
 public:
  OneTBBTaskFunctor(ITaskFunctor* functor,ITask* task)
  : m_functor(functor), m_task(task) {}
 public:
  void operator()() const
  {
    if (m_functor){
      ITaskFunctor* tf = m_functor;
      m_functor = 0;
      TaskContext task_context(m_task);
      //cerr << "FUNC=" << typeid(*tf).name();
      tf->executeFunctor(task_context);
    }
  }
 public:
  mutable ITaskFunctor* m_functor;
  ITask* m_task;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OneTBBTask
: public ITask
{
 public:
  static const int FUNCTOR_CLASS_SIZE = 32;
 public:
  OneTBBTask(ITaskFunctor* f)
  : m_functor(f)
  {
    m_functor = f->clone(functor_buf,FUNCTOR_CLASS_SIZE);
  }
 public:
  OneTBBTaskFunctor taskFunctor() { return OneTBBTaskFunctor(m_functor,this); }
  void launchAndWait() override;
  void launchAndWait(ConstArrayView<ITask*> tasks) override;
 protected:
  virtual ITask* _createChildTask(ITaskFunctor* functor) override;
 public:
  ITaskFunctor* m_functor;
  char functor_buf[FUNCTOR_CLASS_SIZE];
};
using TBBTask = OneTBBTask;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#else // ARCANE_USE_ONETBB

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LegacyTBBTask
: public tbb::task
, public ITask
{
 public:
  static const int FUNCTOR_CLASS_SIZE = 32;
 public:
  LegacyTBBTask(ITaskFunctor* f)
  : m_functor(f)
  {
    m_functor = f->clone(functor_buf,FUNCTOR_CLASS_SIZE);
  }
 public:
  tbb::task* execute() override
  {
    if (m_functor){
      ITaskFunctor* tf = m_functor;
      m_functor = 0;
      TaskContext task_context(this);
      //cerr << "FUNC=" << typeid(*tf).name();
      tf->executeFunctor(task_context);
    }
    return nullptr;
  }
  void launchAndWait() override;
  void launchAndWait(ConstArrayView<ITask*> tasks) override;
 protected:
  ITask* _createChildTask(ITaskFunctor* functor) final;
 public:
  ITaskFunctor* m_functor;
  char functor_buf[FUNCTOR_CLASS_SIZE];
};
using TBBTask = LegacyTBBTask;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_USE_ONETBB

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Ne pas utiliser l'observer locale au task_arena.
 * Utiliser l'observer global au scheduler.
 * Pour l'id, utiliser tbb::this_task_arena::current_thread_index().
 */
class TBBTaskImplementation
: public ITaskImplementation
{
  class Impl;
  class ParallelForExecute;
  template<int RankValue>
  class MDParallelForExecute;

 public:
  // Pour des raisons de performance, s'aligne sur une ligne de cache
  // et utilise un padding.
  class ARCANE_ALIGNAS_PACKED(64) TaskThreadInfo
  {
  public:
    TaskThreadInfo() : m_task_index(-1){}
  public:
    void setTaskIndex(Integer v) { m_task_index = v; }
    Integer taskIndex() const { return m_task_index; }
  private:
    Integer m_task_index;
  };
  /*!
   * \brief Classe pour positionner TaskThreadInfo::taskIndex().
   *
   * Permet de positionner la valeur de TaskThreadInfo::taskIndex()
   * lors de la construction et de remettre la valeur d'avant
   * dans le destructeur.
   */
  class TaskInfoLockGuard
  {
   public:
    TaskInfoLockGuard(TaskThreadInfo* tti,Integer task_index)
    : m_tti(tti), m_old_task_index(-1)
    {
      if (tti){
        m_old_task_index = tti->taskIndex();
        tti->setTaskIndex(task_index);
      }
    }
    ~TaskInfoLockGuard()
    {
      if (m_tti)
        m_tti->setTaskIndex(m_old_task_index);
    }
   private:
    TaskThreadInfo* m_tti;
    Integer m_old_task_index;
  };

 public:
  TBBTaskImplementation(const ServiceBuildInfo& sbi)
  {
  }
  TBBTaskImplementation() = default;
  ~TBBTaskImplementation() override;
 public:
  void build() {}
  void initialize(Int32 nb_thread) override;
  void terminate() override;

  ITask* createRootTask(ITaskFunctor* f) override
  {
#ifdef ARCANE_USE_ONETBB
    OneTBBTask* t = new OneTBBTask(f);
#else
    LegacyTBBTask* t = new(tbb::task::allocate_root()) LegacyTBBTask(f);
#endif
    return t;
  }

  void executeParallelFor(Int32 begin,Int32 size,const ParallelLoopOptions& options,IRangeFunctor* f) final;
  void executeParallelFor(Int32 begin,Int32 size,Integer grain_size,IRangeFunctor* f) final;
  void executeParallelFor(Int32 begin,Int32 size,IRangeFunctor* f) final
  {
    executeParallelFor(begin,size,TaskFactory::defaultParallelLoopOptions(),f);
  }
  void executeParallelFor(const ParallelFor1DLoopInfo& loop_info) override;

  void executeParallelFor(const ComplexForLoopRanges<1>& loop_ranges,
                          const ForLoopRunInfo& run_info,
                          IMDRangeFunctor<1>* functor) final
  {
    _executeMDParallelFor<1>(loop_ranges,functor,run_info);
  }
  void executeParallelFor(const ComplexForLoopRanges<2>& loop_ranges,
                          const ForLoopRunInfo& run_info,
                          IMDRangeFunctor<2>* functor) final
  {
    _executeMDParallelFor<2>(loop_ranges,functor,run_info);
  }
  void executeParallelFor(const ComplexForLoopRanges<3>& loop_ranges,
                          const ForLoopRunInfo& run_info,
                          IMDRangeFunctor<3>* functor) final
  {
    _executeMDParallelFor<3>(loop_ranges,functor,run_info);
  }
  void executeParallelFor(const ComplexForLoopRanges<4>& loop_ranges,
                          const ForLoopRunInfo& run_info,
                          IMDRangeFunctor<4>* functor) final
  {
    _executeMDParallelFor<4>(loop_ranges,functor,run_info);
  }

  bool isActive() const final
  {
    return m_is_active;
  }

  Int32 nbAllowedThread() const final;

  Int32 currentTaskThreadIndex() const final
  {
    return (nbAllowedThread() <= 1 ) ? 0 : _currentTaskTreadIndex();
  }

  Int32 currentTaskIndex() const final;

  void printInfos(std::ostream& o) const final;

 public:

  /*!
   * \brief Instance de \a TaskThreadInfo associé au thread courant.
   *
   * Peut-être nul si le thread courant n'est pas associé à un thread TBB
   * ou si en dehors d'une exécution d'une tâche ou d'une boucle parallèle.
   */
  TaskThreadInfo* currentTaskThreadInfo() const;

 private:

  bool m_is_active = false;
  Impl* m_p = nullptr;

 private:

  template<int RankValue> void
  _executeMDParallelFor(const ComplexForLoopRanges<RankValue>& loop_ranges,
                        IMDRangeFunctor<RankValue>* functor,
                        const ForLoopRunInfo& run_info);
  void _executeParallelFor(const ParallelFor1DLoopInfo& loop_info);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TBBTaskImplementation::Impl
{
  class TaskObserver
  : public tbb::task_scheduler_observer
  {
   public:
    TaskObserver(TBBTaskImplementation::Impl* p)
    :
#ifdef ARCANE_USE_ONETBB
    tbb::task_scheduler_observer(p->m_main_arena),
#endif
    m_p(p)
    {
    }
    void on_scheduler_entry(bool is_worker) override
    {
      m_p->notifyThreadCreated(is_worker);
    }
    void on_scheduler_exit(bool is_worker) override
    {
      m_p->notifyThreadDestroyed(is_worker);
    }
    TBBTaskImplementation::Impl* m_p;
  };

 public:
  Impl() :
  m_task_observer(this),
  m_thread_task_infos(AlignedMemoryAllocator::CacheLine())
  {
#ifdef ARCANE_USE_ONETBB
    m_nb_allowed_thread = tbb::info::default_concurrency();
#else
    m_nb_allowed_thread = tbb::task_scheduler_init::default_num_threads();
#endif
    _init();
  }
  Impl(Int32 nb_thread)
  :
#ifndef ARCANE_USE_ONETBB
  m_scheduler_init(nb_thread),
#endif
  m_main_arena(nb_thread),
  m_task_observer(this),
  m_thread_task_infos(AlignedMemoryAllocator::CacheLine())
  {
    m_nb_allowed_thread = nb_thread;
    _init();
  }
 public:
  Int32 nbAllowedThread() const { return m_nb_allowed_thread; }
  TaskThreadInfo* threadTaskInfo(Integer index) { return &m_thread_task_infos[index]; }
 private:
  Int32 m_nb_allowed_thread;

 public:
  void terminate()
  {
    for( auto x : m_sub_arena_list ){
      if (x)
        x->terminate();
      delete x;
    }
    m_sub_arena_list.clear();
    m_main_arena.terminate();
#ifdef ARCANE_USE_ONETBB
    m_task_observer.observe(false);
    oneapi::tbb::finalize(m_task_scheduler_handle);
#else
    m_scheduler_init.terminate();
    m_task_observer.observe(false);
#endif
  }
 public:
  void notifyThreadCreated(bool is_worker)
  {
    std::thread::id my_thread_id = std::this_thread::get_id();

#ifdef ARCANE_USE_ONETBB
    // Avec OneTBB, cette méthode est appelée à chaque fois qu'on rentre
    // dans notre 'task_arena'. Comme il ne faut appeler qu'une seule
    // fois la méthode de notification on utilise un ensemble pour
    // conserver la liste des threads déjà créés.
    // NOTE: On ne peut pas utiliser cette méthode avec la version TBB historique
    // (2018) car cette méthode 'contains' n'existe pas
    if (m_constructed_thread_map.contains(my_thread_id))
      return;
    m_constructed_thread_map.insert(my_thread_id);
#endif

    {
      if (TaskFactory::verboseLevel()>=1){
        std::ostringstream ostr;
        ostr << "TBB: CREATE THREAD"
             << " nb_allowed=" << m_nb_allowed_thread
#ifdef ARCANE_USE_ONETBB
             << " tbb_default_allowed=" << tbb::info::default_concurrency()
#else
             << " tbb_default_allowed=" << tbb::task_scheduler_init::default_num_threads()
#endif
             << " id=" << my_thread_id
             << " arena_id=" << _currentTaskTreadIndex()
             << " is_worker=" << is_worker
             << "\n";
        std::cout << ostr.str();
      }
      TaskFactoryInternal::notifyThreadCreated();
    }
  }

  void notifyThreadDestroyed([[maybe_unused]] bool is_worker)
  {
#ifdef ARCANE_USE_ONETBB
    // Avec OneTBB, cette méthode est appelée à chaque fois qu'on sort
    // de l'arène principale. Du coup elle ne correspond pas vraiment à une
    // destruction de thread. On ne fait donc rien pour cette notification
    // TODO: Regarder comment on peut être notifié de la destruction effective
    // du thread.
#else
    // Il faut toujours un verrou car on n'est pas certain que
    // les méthodes appelées par l'observable soient thread-safe
    // (et aussi TaskFactory::createThreadObservable() ne l'est pas)
    std::scoped_lock sl(m_thread_created_mutex);
    if (TaskFactory::verboseLevel()>=1){
      std::cout << "TBB: DESTROY THREAD"
                << " id=" << std::this_thread::get_id()
                << " arena_id=" << _currentTaskTreadIndex()
                << " is_worker=" << is_worker
                << '\n';
    }
    // TODO: jamais utilisé. Sera supprimé au passage à OneTBB.
    TaskFactory::destroyThreadObservable()->notifyAllObservers();
#endif
  }
 private:
#ifdef ARCANE_USE_ONETBB
#if TBB_VERSION_MAJOR>2021 || (TBB_VERSION_MAJOR==2021 && TBB_VERSION_MINOR>5)
  oneapi::tbb::task_scheduler_handle m_task_scheduler_handle = oneapi::tbb::attach();
#else
  oneapi::tbb::task_scheduler_handle m_task_scheduler_handle = tbb::task_scheduler_handle::get();
#endif
#else
  tbb::task_scheduler_init m_scheduler_init;
#endif
 public:
  tbb::task_arena m_main_arena;
  //! Tableau dont le i-ème élément contient la tbb::task_arena pour \a i thread.
  UniqueArray<tbb::task_arena*> m_sub_arena_list;
 private:
  TaskObserver m_task_observer;
  std::mutex m_thread_created_mutex;
  UniqueArray<TaskThreadInfo> m_thread_task_infos;
#ifdef ARCANE_USE_ONETBB
  tbb::concurrent_set<std::thread::id> m_constructed_thread_map;
#endif
  void _init()
  {
    if (TaskFactory::verboseLevel()>=1){
      std::cout << "TBB: TBBTaskImplementationInit nb_allowed_thread=" << m_nb_allowed_thread
                << " id=" << std::this_thread::get_id()
                << " version=" << TBB_VERSION_MAJOR << "." << TBB_VERSION_MINOR
                << "\n";
    }
    m_thread_task_infos.resize(m_nb_allowed_thread);
    m_task_observer.observe(true);
    Integer max_arena_size = m_nb_allowed_thread;
    // Limite artificiellement le nombre de tbb::task_arena
    // pour éviter d'avoir trop d'objets alloués.
    if (max_arena_size>512)
      max_arena_size = 512;
    if (max_arena_size<2)
      max_arena_size = 2;
    m_sub_arena_list.resize(max_arena_size);
    m_sub_arena_list[0] = m_sub_arena_list[1] = nullptr;
    for( Integer i=2; i<max_arena_size; ++i )
      m_sub_arena_list[i] = new tbb::task_arena(i);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exécuteur pour une boucle 1D.
 */
class TBBParallelFor
{
 public:
  TBBParallelFor(IRangeFunctor* f,Int32 nb_allowed_thread,ForLoopOneExecStat* stat_info)
  : m_functor(f), m_stat_info(stat_info), m_nb_allowed_thread(nb_allowed_thread){}
 public:

  void operator()(tbb::blocked_range<Integer>& range) const
  {
#ifdef ARCANE_CHECK
    if (TaskFactory::verboseLevel()>=3){
      std::ostringstream o;
      o << "TBB: INDEX=" << TaskFactory::currentTaskThreadIndex()
        << " id=" << std::this_thread::get_id()
        << " max_allowed=" << m_nb_allowed_thread
        << " range_begin=" << range.begin() << " range_size=" << range.size()
        << "\n";
      std::cout << o.str();
      std::cout.flush();
    }

    int tbb_index = _currentTaskTreadIndex();
    if (tbb_index<0 || tbb_index>=m_nb_allowed_thread)
      ARCANE_FATAL("Invalid index for thread idx={0} valid_interval=[0..{1}[",
                   tbb_index,m_nb_allowed_thread);
#endif

    if (m_stat_info)
      m_stat_info->incrementNbChunk();
    m_functor->executeFunctor(range.begin(),CheckedConvert::toInteger(range.size()));
  }

 private:
  IRangeFunctor* m_functor;
  ForLoopOneExecStat* m_stat_info = nullptr;
  Int32 m_nb_allowed_thread;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exécuteur pour une boucle multi-dimension.
 */
template<int RankValue>
class TBBMDParallelFor
{
 public:

  TBBMDParallelFor(IMDRangeFunctor<RankValue>* f,Int32 nb_allowed_thread,ForLoopOneExecStat* stat_info)
  : m_functor(f), m_stat_info(stat_info), m_nb_allowed_thread(nb_allowed_thread){}

 public:

  void operator()(blocked_nd_range<Int32, RankValue>& range) const
  {
#ifdef ARCANE_CHECK
    if (TaskFactory::verboseLevel()>=3){
      std::ostringstream o;
      o << "TBB: INDEX=" << TaskFactory::currentTaskThreadIndex()
        << " id=" << std::this_thread::get_id()
        << " max_allowed=" << m_nb_allowed_thread
        << " MDFor ";
      for( Int32 i=0; i<RankValue; ++i ){
        Int32 r0 = static_cast<Int32>(range.dim(i).begin());
        Int32 r1 = static_cast<Int32>(range.dim(i).size());
        o << " range" << i << " (begin=" << r0 << " size=" << r1 << ")";
      }
      o << "\n";
      std::cout << o.str();
      std::cout.flush();
    }

    int tbb_index = _currentTaskTreadIndex();
    if (tbb_index<0 || tbb_index>=m_nb_allowed_thread)
      ARCANE_FATAL("Invalid index for thread idx={0} valid_interval=[0..{1}[",
                   tbb_index,m_nb_allowed_thread);
#endif

    if (m_stat_info)
      m_stat_info->incrementNbChunk();
    m_functor->executeFunctor(_fromTBBRange(range));
  }

 private:

  IMDRangeFunctor<RankValue>* m_functor = nullptr;
  ForLoopOneExecStat* m_stat_info = nullptr;
  Int32 m_nb_allowed_thread;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation déterministe de ParallelFor.
 *
 * L'implémentation est déterministe dans le sens où elle ne dépend que
 * de l'intervalle d'itération (m_begin_index et m_size),
 * du nombre de threads spécifié (\a m_nb_thread) et de la taille du grain (\a m_grain_size).
 *
 * L'algorithme utilisé se rapproche de celui utilisé par OpenMP pour un
 * parallel for avec l'option statique: on découpe l'intervalle d'itération
 * en plusieurs blocs et chaque bloc est assigné à une tâche en fonction
 * d'un algorithme round-robin.
 * Pour déterminer le nombre de blocs, deux cas sont possibles:
 * - si \a m_grain_size n'est pas spécifié, on découpe l'intervalle
 * d'itération en un nombre de blocs équivalent au nombre de threads utilisés.
 * - si \a m_grain_size est spécifié, le nombre de blocs sera égal
 * à \a m_size divisé par \a m_grain_size.
 */
class TBBDeterministicParallelFor
{
 public:
  TBBDeterministicParallelFor(TBBTaskImplementation* impl,const TBBParallelFor& tbb_for,
                              Integer begin_index,Integer size,Integer grain_size,Integer nb_thread)
  : m_impl(impl), m_tbb_for(tbb_for), m_nb_thread(nb_thread), m_begin_index(begin_index), m_size(size),
    m_grain_size(grain_size), m_nb_block(0), m_block_size(0), m_nb_block_per_thread(0)
  {
    if (m_nb_thread<1)
      m_nb_thread = 1;

    if (m_grain_size>0){
      m_block_size = m_grain_size;
      if (m_block_size>0){
        m_nb_block = m_size / m_block_size;
        if ((m_size % m_block_size)!=0)
          ++m_nb_block;
      }
      else
        m_nb_block = 1;
      m_nb_block_per_thread = m_nb_block / m_nb_thread;
      if ((m_nb_block % m_nb_thread) != 0)
        ++m_nb_block_per_thread;
    }
    else{
      if (m_nb_block<1)
        m_nb_block = m_nb_thread;
      m_block_size = m_size / m_nb_block;
      m_nb_block_per_thread = 1;
    }
    if (TaskFactory::verboseLevel()>=2){
      std::cout << "TBBDeterministicParallelFor: BEGIN=" << m_begin_index << " size=" << m_size
                << " grain_size=" << m_grain_size
                << " nb_block=" << m_nb_block << " nb_thread=" << m_nb_thread
                << " nb_block_per_thread=" << m_nb_block_per_thread
                << " block_size=" << m_block_size
                << " block_size*nb_block=" << m_block_size*m_nb_block << '\n';
    }
  }
 public:

  /*!
   * \brief Opérateur pour un thread donné.
   *
   * En règle générale, range.size() vaudra 1 car un thread ne traitera qu'une itération
   * mais ce n'est a priori pas garanti par les TBB.
   */
  void operator()(tbb::blocked_range<Integer>& range) const
  {
    Integer nb_iter = static_cast<Integer>(range.size());
    for( Integer i=0; i<nb_iter; ++i ){
      Integer task_id = range.begin() + i;
      for ( Integer k=0, kn=m_nb_block_per_thread; k<kn; ++k ){
        Integer block_id = task_id + (k * m_nb_thread);
        if (block_id<m_nb_block)
          _doBlock(task_id,block_id);
      }
    }
  }

  void _doBlock(Integer task_id,Integer block_id) const
  {
    TBBTaskImplementation::TaskInfoLockGuard guard(m_impl->currentTaskThreadInfo(),task_id);

    Integer iter_begin = block_id * m_block_size;
    Integer iter_size = m_block_size;
    if ((block_id+1)==m_nb_block){
      // Pour le dernier bloc, la taille est le nombre d'éléments restants
      iter_size = m_size - iter_begin;
    }
    iter_begin += m_begin_index;
#ifdef ARCANE_CHECK
    if (TaskFactory::verboseLevel()>=3){
      std::ostringstream o;
      o << "TBB: DoBlock: BLOCK task_id=" << task_id << " block_id=" << block_id
        << " iter_begin=" << iter_begin << " iter_size=" << iter_size << '\n';
      std::cout << o.str();
      std::cout.flush();
    }
#endif
    if (iter_size>0){
      auto r = tbb::blocked_range<int>(iter_begin,iter_begin + iter_size);
      m_tbb_for(r);
    }
  }

 private:

  TBBTaskImplementation* m_impl;
  const TBBParallelFor& m_tbb_for;
  Integer m_nb_thread;
  Integer m_begin_index;
  Integer m_size;
  Integer m_grain_size;
  Integer m_nb_block;
  Integer m_block_size;
  Integer m_nb_block_per_thread;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TBBTaskImplementation::ParallelForExecute
{
 public:

  ParallelForExecute(TBBTaskImplementation* impl, const ParallelLoopOptions& options,
                     Integer begin,Integer size,IRangeFunctor* f,ForLoopOneExecStat* stat_info)
  : m_impl(impl), m_begin(begin), m_size(size), m_functor(f), m_options(options), m_stat_info(stat_info){}

 public:

  void operator()() const
  {
    Integer nb_thread = m_options.maxThread();
    TBBParallelFor pf(m_functor,nb_thread,m_stat_info);
    Integer gsize = m_options.grainSize();
    tbb::blocked_range<Integer> range(m_begin,m_begin+m_size);
    if (TaskFactory::verboseLevel()>=1)
      std::cout << "TBB: TBBTaskImplementationInit ParallelForExecute begin=" << m_begin
                << " size=" << m_size << " gsize=" << gsize
                << " partitioner=" << (int)m_options.partitioner()
                << " nb_thread=" << nb_thread
                << " has_stat_info=" << (m_stat_info!=nullptr)
                << '\n';

    if (gsize>0)
      range = tbb::blocked_range<Integer>(m_begin,m_begin+m_size,gsize);

    if (m_options.partitioner()==ParallelLoopOptions::Partitioner::Static){
      tbb::parallel_for(range,pf,tbb::static_partitioner());
    }
    else if (m_options.partitioner()==ParallelLoopOptions::Partitioner::Deterministic){
      tbb::blocked_range<Integer> range2(0,nb_thread,1);
      TBBDeterministicParallelFor dpf(m_impl,pf,m_begin,m_size,gsize,nb_thread);
      tbb::parallel_for(range2,dpf);
    }
    else
      tbb::parallel_for(range,pf);
  }
 private:
  TBBTaskImplementation* m_impl = nullptr;
  Integer m_begin;
  Integer m_size;
  IRangeFunctor* m_functor = nullptr;
  ParallelLoopOptions m_options;
  ForLoopOneExecStat* m_stat_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<int RankValue>
class TBBTaskImplementation::MDParallelForExecute
{
 public:

  MDParallelForExecute(TBBTaskImplementation* impl,
                       const ParallelLoopOptions& options,
                       const ComplexForLoopRanges<RankValue>& range,
                       IMDRangeFunctor<RankValue>* f,[[maybe_unused]] ForLoopOneExecStat* stat_info)
  : m_impl(impl)
  , m_tbb_range(_toTBBRange(range))
  , m_functor(f)
  , m_options(options)
  , m_stat_info(stat_info)
  {
    // On ne peut pas modifier les valeurs d'une instance de tbb::blocked_rangeNd.
    // Il faut donc en reconstruire une complètement.
    FixedArray<size_t,RankValue> all_grain_sizes;
    Int32 gsize = m_options.grainSize();
    if (gsize>0){
      // Si la taille du grain est différent zéro, il faut la répartir
      // sur l'ensemble des dimensions. On commence par la dernière.
      // TODO: regarder pourquoi dans certains cas les performances sont
      // inférieures à celles qu'on obtient en utilisant un partitionneur
      // statique.
      constexpr bool is_verbose = false;
      std::array<Int32,RankValue> range_extents = range.extents().asStdArray();
      double ratio = static_cast<double>(gsize) / static_cast<double>(range.nbElement());
      if constexpr (is_verbose){
        std::cout << "GSIZE=" << gsize << " rank=" << RankValue << " ratio=" << ratio;
        for(Int32 i=0; i<RankValue; ++i )
          std::cout << " range" << i << "=" << range_extents[i];
        std::cout << "\n";
      }
      Int32 index = RankValue - 1;
      Int32 remaining_grain = gsize;
      for( ; index>=0; --index ){
        Int32 current = range_extents[index];
        if constexpr (is_verbose)
          std::cout << "Check index=" << index << " remaining=" << remaining_grain << " current=" << current << "\n";
        if (remaining_grain>current){
          all_grain_sizes[index] = current;
          remaining_grain /= current;
        }
        else{
          all_grain_sizes[index] = remaining_grain;
          break;
        }
      }
      for( Int32 i=0; i<index; ++i )
        all_grain_sizes[i] = 1;
      if constexpr (is_verbose){
        for(Int32 i=0; i<RankValue; ++i )
          std::cout << " grain" << i << "=" << all_grain_sizes[i];
        std::cout << "\n";
      }
      m_tbb_range = _toTBBRangeWithGrain(m_tbb_range,all_grain_sizes);
    }
  }

 public:

  void operator()() const
  {
    Integer nb_thread = m_options.maxThread();
    TBBMDParallelFor<RankValue> pf(m_functor,nb_thread,m_stat_info);

    if (m_options.partitioner()==ParallelLoopOptions::Partitioner::Static){
      tbb::parallel_for(m_tbb_range,pf,tbb::static_partitioner());
    }
    else if (m_options.partitioner()==ParallelLoopOptions::Partitioner::Deterministic){
      // TODO: implémenter le mode déterministe
      ARCANE_THROW(NotImplementedException,"ParallelLoopOptions::Partitioner::Deterministic for multi-dimensionnal loops");
      //tbb::blocked_range<Integer> range2(0,nb_thread,1);
      //TBBDeterministicParallelFor dpf(m_impl,pf,m_begin,m_size,gsize,nb_thread);
      //tbb::parallel_for(range2,dpf);
    }
    else{
      tbb::parallel_for(m_tbb_range,pf);
    }
  }
 private:
  TBBTaskImplementation* m_impl = nullptr;
  blocked_nd_range<Int32, RankValue> m_tbb_range;
  IMDRangeFunctor<RankValue>* m_functor = nullptr;
  ParallelLoopOptions m_options;
  ForLoopOneExecStat* m_stat_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TBBTaskImplementation::
~TBBTaskImplementation()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TBBTaskImplementation::
initialize(Int32 nb_thread)
{
  if (nb_thread<0)
    nb_thread = 0;
  m_is_active = (nb_thread!=1);
  if (nb_thread!=0)
    m_p = new Impl(nb_thread);
  else
    m_p = new Impl();
  ParallelLoopOptions opts = TaskFactory::defaultParallelLoopOptions();
  opts.setMaxThread(nbAllowedThread());
  TaskFactory::setDefaultParallelLoopOptions(opts);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TBBTaskImplementation::
terminate()
{
  m_p->terminate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 TBBTaskImplementation::
nbAllowedThread() const
{
  return m_p->nbAllowedThread();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TBBTaskImplementation::
printInfos(std::ostream& o) const
{
#ifdef ARCANE_USE_ONETBB
  o << "OneTBBTaskImplementation"
    << " version=" << TBB_VERSION_STRING
    << " interface=" << TBB_INTERFACE_VERSION
    << " runtime_interface=" << TBB_runtime_interface_version();
#else
  o << "TBBTaskImplementation"
    << " version=" << TBB_VERSION_MAJOR << "." << TBB_VERSION_MINOR
    << " interface=" << TBB_INTERFACE_VERSION;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TBBTaskImplementation::
_executeParallelFor(const ParallelFor1DLoopInfo& loop_info)
{
  ScopedExecInfo sei(loop_info.runInfo());
  ForLoopOneExecStat* stat_info = sei.statInfo();
  impl::ScopedStatLoop scoped_loop(sei.isOwn() ? stat_info : nullptr);

  Int32 begin = loop_info.beginIndex();
  Int32 size = loop_info.size();
  ParallelLoopOptions options = loop_info.runInfo().options().value_or(TaskFactory::defaultParallelLoopOptions());
  IRangeFunctor* f = loop_info.functor();

  Integer max_thread = options.maxThread();
  Integer nb_allowed_thread = m_p->nbAllowedThread();
  if (max_thread<0)
    max_thread = nb_allowed_thread;

  if (TaskFactory::verboseLevel()>=1)
    std::cout << "TBB: TBBTaskImplementation executeParallelFor begin=" << begin
              << " size=" << size << " max_thread=" << max_thread
              << " grain_size=" << options.grainSize()
              << " nb_allowed=" << nb_allowed_thread << '\n';

  // En exécution séquentielle, appelle directement la méthode \a f.
  if (max_thread==1 || max_thread==0){
    f->executeFunctor(begin,size);
    return;
  }

  // Remplace les valeurs non initialisées de \a options par celles de \a m_default_loop_options
  ParallelLoopOptions true_options(options);
  true_options.mergeUnsetValues(TaskFactory::defaultParallelLoopOptions());
  true_options.setMaxThread(max_thread);

  ParallelForExecute pfe(this,true_options,begin,size,f,stat_info);

  tbb::task_arena* used_arena = nullptr;
  if (max_thread<nb_allowed_thread && max_thread<m_p->m_sub_arena_list.size())
    used_arena = m_p->m_sub_arena_list[max_thread];
  if (!used_arena)
    used_arena = &(m_p->m_main_arena);
  used_arena->execute(pfe);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TBBTaskImplementation::
executeParallelFor(const ParallelFor1DLoopInfo& loop_info)
{
  _executeParallelFor(loop_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exécution d'une boucle N-dimensions.
 *
 * \warning L'implémentation actuelle ne tient pas compte de \a options
 * pour les boucles autres que une dimension.
 */
template<int RankValue> void TBBTaskImplementation::
_executeMDParallelFor(const ComplexForLoopRanges<RankValue>& loop_ranges,
                      IMDRangeFunctor<RankValue>* functor,
                      const ForLoopRunInfo& run_info)
{
  ParallelLoopOptions options;
  if (run_info.options().has_value())
    options = run_info.options().value();

  ScopedExecInfo sei(run_info);
  ForLoopOneExecStat* stat_info = sei.statInfo();
  impl::ScopedStatLoop scoped_loop(sei.isOwn() ? stat_info : nullptr);

  if (TaskFactory::verboseLevel()>=1){
    std::cout << "TBB: TBBTaskImplementation executeMDParallelFor nb_dim=" << RankValue
              << " nb_element=" << loop_ranges.nbElement()
              << " grain_size=" << options.grainSize()
              << " name=" << run_info.traceInfo().traceInfo()
              << " has_stat_info=" << (stat_info!=nullptr)
              << '\n';
  }

  Integer max_thread = options.maxThread();
  // En exécution séquentielle, appelle directement la méthode \a f.
  if (max_thread==1 || max_thread==0){
    functor->executeFunctor(loop_ranges);
    return;
  }

  // Remplace les valeurs non initialisées de \a options par celles de \a m_default_loop_options
  ParallelLoopOptions true_options(options);
  true_options.mergeUnsetValues(TaskFactory::defaultParallelLoopOptions());

  Integer nb_allowed_thread = m_p->nbAllowedThread();
  if (max_thread<0)
    max_thread = nb_allowed_thread;
  tbb::task_arena* used_arena = nullptr;
  if (max_thread<nb_allowed_thread)
    used_arena = m_p->m_sub_arena_list[max_thread];
  if (!used_arena)
    used_arena = &(m_p->m_main_arena);

  // Pour l'instant pour la dimension 1, utilise le 'ParallelForExecute' historique
  if constexpr (RankValue==1){
    auto range_1d = _toTBBRange(loop_ranges);
    auto x1 = [&](Integer begin,Integer size)
              {
                functor->executeFunctor(makeLoopRanges(ForLoopRange(begin,size)));
                //functor->executeFunctor(ComplexForLoopRanges<1>(begin,size));
              };
    LambdaRangeFunctorT<decltype(x1)> functor_1d(x1);
    Integer begin1 = CheckedConvert::toInteger(range_1d.dim(0).begin());
    Integer size1 = CheckedConvert::toInteger(range_1d.dim(0).size());
    ParallelForExecute pfe(this,true_options,begin1,size1,&functor_1d,stat_info);
    used_arena->execute(pfe);
  }
  else{
    MDParallelForExecute<RankValue> pfe(this,true_options,loop_ranges,functor,stat_info);
    used_arena->execute(pfe);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TBBTaskImplementation::
executeParallelFor(Integer begin,Integer size,Integer grain_size,IRangeFunctor* f)
{
  ParallelLoopOptions opts(TaskFactory::defaultParallelLoopOptions());
  opts.setGrainSize(grain_size);
  ForLoopRunInfo run_info(opts);
  executeParallelFor(ParallelFor1DLoopInfo(begin,size,f,run_info));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TBBTaskImplementation::
executeParallelFor(Integer begin,Integer size,const ParallelLoopOptions& options,IRangeFunctor* f)
{
  executeParallelFor(ParallelFor1DLoopInfo(begin,size,f,ForLoopRunInfo(options)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TBBTaskImplementation::TaskThreadInfo* TBBTaskImplementation::
currentTaskThreadInfo() const
{
  Int32 thread_id = currentTaskThreadIndex();
  if (thread_id>=0)
    return m_p->threadTaskInfo(thread_id);
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 TBBTaskImplementation::
currentTaskIndex() const
{
  Int32 thread_id = currentTaskThreadIndex();
#ifdef ARCANE_USE_ONETBB
  if (thread_id<0 || thread_id>=m_p->nbAllowedThread())
    return 0;
#endif
  TBBTaskImplementation::TaskThreadInfo* tti = currentTaskThreadInfo();
  if (tti){
    Int32 task_index = tti->taskIndex();
    if (task_index>=0)
      return task_index;
  }
  return thread_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_USE_ONETBB

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneTBBTask::
launchAndWait()
{
  tbb::task_group task_group;
  task_group.run(taskFunctor());
  task_group.wait();
  delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneTBBTask::
launchAndWait(ConstArrayView<ITask*> tasks)
{
  tbb::task_group task_group;
  Integer n = tasks.size();
  if (n==0)
    return;

  //set_ref_count(n+1);
  for( Integer i=0; i<n; ++i ){
    OneTBBTask* t = static_cast<OneTBBTask*>(tasks[i]);
    task_group.run(t->taskFunctor());
  }
  task_group.wait();
  for( Integer i=0; i<n; ++i ){
    OneTBBTask* t = static_cast<OneTBBTask*>(tasks[i]);
    delete t;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITask* OneTBBTask::
_createChildTask(ITaskFunctor* functor)
{
  OneTBBTask* t = new OneTBBTask(functor);
  return t;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#else // ARCANE_USE_ONETBB

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LegacyTBBTask::
launchAndWait()
{
  task::spawn_root_and_wait(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LegacyTBBTask::
launchAndWait(ConstArrayView<ITask*> tasks)
{
  Integer n = tasks.size();
  if (n==0)
    return;

  set_ref_count(n+1);
  for( Integer i=0; i<n-1; ++i ){
    TBBTask* t = static_cast<TBBTask*>(tasks[i]);
    spawn(*t);
  }
  spawn_and_wait_for_all(*static_cast<TBBTask*>(tasks[n-1]));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITask* LegacyTBBTask::
_createChildTask(ITaskFunctor* functor)
{
  TBBTask* t = new(allocate_child()) TBBTask(functor);
  return t;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_USE_ONETBB

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: a supprimer maintenant qu'on utilise 'DependencyInjection'
ARCANE_REGISTER_APPLICATION_FACTORY(TBBTaskImplementation,ITaskImplementation,
                                    TBBTaskImplementation);

ARCANE_DI_REGISTER_PROVIDER(TBBTaskImplementation,
                            DependencyInjection::ProviderProperty("TBBTaskImplementation"),
                            ARCANE_DI_INTERFACES(ITaskImplementation),
                            ARCANE_DI_EMPTY_CONSTRUCTOR());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
