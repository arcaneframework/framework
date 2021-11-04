// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* OneTBBTaskImplementation.cc                                 (C) 2000-2021 */
/*                                                                           */
/* Implémentation des tâches utilisant OneTBB version 2021+.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IThreadImplementation.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/IFunctor.h"
#include "arcane/utils/Mutex.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/IObservable.h"
#include "arcane/FactoryService.h"
#include "arcane/Concurrency.h"

// Nécessaire pour avoir accès à task_scheduler_handle
#define TBB_PREVIEW_WAITING_FOR_WORKERS 1
#include <tbb/tbb.h>
#include <oneapi/tbb/global_control.h>

#include <new>
#include <stack>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
// TODO: utiliser un pool mémoire spécifique pour gérer les
// OneTBBTask pour optimiser les new/delete des instances de cette classe.
// Auparavant avec les anciennes versions de TBB cela était géré avec
// la méthode 'tbb::task::allocate_child()'.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OneTBBTaskImplementation;

namespace
{
inline int _currentTaskTreadIndex()
{
  // NOTE: Avec OneTBB 2021, la valeur n'est plus '0' si on appelle cette méthode
  // depuis un thread en dehors d'un task_arena. Avec la version 2021,
  // la valeur est 65535.
  return tbb::this_task_arena::current_thread_index();
}
}

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Ne pas utiliser l'observer locale au task_arena.
 * Utiliser l'observer global au scheduler.
 * Pour l'id, utiliser tbb::this_task_arena::current_thread_index().
 */
class OneTBBTaskImplementation
: public ITaskImplementation
{
  class Impl;
  class ParallelForExecute;

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
    : m_tti(tti), m_task_index(task_index), m_old_task_index(-1)
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
    Integer m_task_index;
    Integer m_old_task_index;
  };

 public:
  OneTBBTaskImplementation(const ServiceBuildInfo& sbi)
  : m_is_active(false), m_p(nullptr)
  {
    ARCANE_UNUSED(sbi);
  }
  ~OneTBBTaskImplementation() override;
 public:
  void build() {}
  void initialize(Int32 nb_thread) override;
  void terminate() override;

  ITask* createRootTask(ITaskFunctor* f) override
  {
    OneTBBTask* t = new OneTBBTask(f);
    return t;
  }

  void executeParallelFor(Integer begin,Integer size,const ParallelLoopOptions& options,IRangeFunctor* f) final;
  void executeParallelFor(Integer begin,Integer size,Integer grain_size,IRangeFunctor* f);
  void executeParallelFor(Integer begin,Integer size,IRangeFunctor* f) final
  {
    executeParallelFor(begin,size,m_default_loop_options,f);
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
  void setDefaultParallelLoopOptions(const ParallelLoopOptions& v) final
  {
    m_default_loop_options = v;
  }

  const ParallelLoopOptions& defaultParallelLoopOptions() final
  {
    return m_default_loop_options;
  }

  void printInfos(ostream& o) const final;

 public:

  /*!
   * \brief Instance de \a TaskThreadInfo associé au thread courant.
   *
   * Peut-être nul si le thread courant n'est pas associé à un thread TBB
   * ou si en dehors d'une exécution d'une tâche ou d'une boucle parallèle.
   */
  TaskThreadInfo* currentTaskThreadInfo() const;

 private:

  bool m_is_active;
  Impl* m_p;
  ParallelLoopOptions m_default_loop_options;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OneTBBTaskImplementation::Impl
{
  class TaskObserver
  : public tbb::task_scheduler_observer
  {
   public:
    TaskObserver(OneTBBTaskImplementation::Impl* p)
    : tbb::task_scheduler_observer(p->m_main_arena), m_p(p)
    {
    }
    void on_scheduler_entry(bool is_worker) override
    {
      ARCANE_UNUSED(is_worker);
      m_p->notifyThreadCreated();
    }
    void on_scheduler_exit(bool is_worker) override
    {
      ARCANE_UNUSED(is_worker);
      m_p->notifyThreadDestroyed();
    }
    OneTBBTaskImplementation::Impl* m_p;
  };

 public:
  Impl()
  : m_task_scheduler_handle(tbb::task_scheduler_handle::get()),
    m_task_observer(this),
    m_thread_task_infos(AlignedMemoryAllocator::CacheLine())
  {
    m_nb_allowed_thread = tbb::info::default_concurrency();
    _init();
  }
  Impl(Int32 nb_thread)
  : m_task_scheduler_handle(tbb::task_scheduler_handle::get()),
    m_main_arena(nb_thread),
    m_task_observer(this), m_thread_task_infos(AlignedMemoryAllocator::CacheLine())
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
    m_task_observer.observe(false);
    oneapi::tbb::finalize(m_task_scheduler_handle);
  }
 public:
  void notifyThreadCreated()
  {
    // Il faut toujours un verrou car on n'est pas certain que
    // les méthodes appelées par l'observable soient thread-safe
    // (et aussi TaskFactory::createThreadObservable() ne l'est pas)
    {
      std::scoped_lock sl(m_thread_created_mutex);
      if (TaskFactory::verboseLevel()>=1){
        std::cout << "TBB: CREATE THREAD"
                  << " nb_allowed=" << m_nb_allowed_thread
                  << " tbb_default_allowed=" << tbb::info::default_concurrency()
                  << " id=" << std::this_thread::get_id()
                  << " arena_id=" << _currentTaskTreadIndex()
                  << "\n";
      }
      TaskFactory::createThreadObservable()->notifyAllObservers();
    }
  }

  void notifyThreadDestroyed()
  {
    // Il faut toujours un verrou car on n'est pas certain que
    // les méthodes appelées par l'observable soient thread-safe
    // (et aussi TaskFactory::createThreadObservable() ne l'est pas)
    std::scoped_lock sl(m_thread_created_mutex);
    if (TaskFactory::verboseLevel()>=1){
      std::cout << "TBB: DESTROY THREAD"
                << " id=" << std::this_thread::get_id()
                << " arena_id=" << _currentTaskTreadIndex()
                << '\n';
    }
    TaskFactory::destroyThreadObservable()->notifyAllObservers();
  }
 private:
  oneapi::tbb::task_scheduler_handle m_task_scheduler_handle;
 public:
  tbb::task_arena m_main_arena;
  //! Tableau dont le i-ème élément contient la tbb::task_arena pour \a i thread.
  std::vector<tbb::task_arena*> m_sub_arena_list;
 private:
  TaskObserver m_task_observer;
  std::mutex m_thread_created_mutex;
  UniqueArray<TaskThreadInfo> m_thread_task_infos;

  void _init()
  {
    if (TaskFactory::verboseLevel()>=1){
      std::cout << "TBB: OneTBBTaskImplementationInit nb_allowed_thread=" << m_nb_allowed_thread
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
    m_sub_arena_list.resize(max_arena_size);
    m_sub_arena_list[0] = m_sub_arena_list[1] = nullptr;
    for( Integer i=2; i<max_arena_size; ++i )
      m_sub_arena_list[i] = new tbb::task_arena(i);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TBBParallelFor
{
 public:
  TBBParallelFor(IRangeFunctor* f,Int32 nb_allowed_thread)
  : m_functor(f), m_nb_allowed_thread(nb_allowed_thread){}
 public:

  void operator()(tbb::blocked_range<Integer>& range) const
  {
#ifdef ARCANE_CHECK
    if (TaskFactory::verboseLevel()>=3){
      ostringstream o;
      o << "TBB: INDEX=" << TaskFactory::currentTaskThreadIndex()
        << " id=" << std::this_thread::get_id()
        << " MAX_ALLOWED=" << m_nb_allowed_thread
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

    m_functor->executeFunctor(range.begin(),CheckedConvert::toInteger(range.size()));
  }

 private:
  IRangeFunctor* m_functor;
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
 * - si \a m_grain_size n'est pas spécifié, on découpe le l'intervalle
 * d'itération en un nombre de blocs équivalent au nombre de threads utilisés.
 * - si \a m_grain_size est spécifié, le nombre de blocs sera égal
 * à \a m_size divisé par \a m_grain_size.
 */
class TBBDeterministicParallelFor
{
 public:
  TBBDeterministicParallelFor(OneTBBTaskImplementation* impl,const TBBParallelFor& tbb_for,
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
    Integer nb_iter = range.size();
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
    OneTBBTaskImplementation::TaskInfoLockGuard guard(m_impl->currentTaskThreadInfo(),task_id);

    Integer iter_begin = block_id * m_block_size;
    Integer iter_size = m_block_size;
    if ((block_id+1)==m_nb_block){
      // Pour le dernier bloc, la taille est le nombre d'éléments restants
      iter_size = m_size - iter_begin;
    }
    iter_begin += m_begin_index;
#ifdef ARCANE_CHECK
    if (TaskFactory::verboseLevel()>=3){
      ostringstream o;
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

  OneTBBTaskImplementation* m_impl;
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

class OneTBBTaskImplementation::ParallelForExecute
{
 public:
  ParallelForExecute(OneTBBTaskImplementation* impl,
                     const ParallelLoopOptions& options,
                     Integer begin,Integer size,IRangeFunctor* f)
  : m_impl(impl), m_begin(begin), m_size(size), m_functor(f), m_options(options){}
 public:
  void operator()() const
  {
    Integer nb_thread = m_options.maxThread();
    TBBParallelFor pf(m_functor,nb_thread);
    Integer gsize = m_options.grainSize();
    tbb::blocked_range<Integer> range(m_begin,m_begin+m_size);
    if (TaskFactory::verboseLevel()>=1)
      std::cout << "TBB: OneTBBTaskImplementationInit ParallelForExemple begin=" << m_begin
                << " size=" << m_size << " gsize=" << gsize << '\n';
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
  OneTBBTaskImplementation* m_impl;
  Integer m_begin;
  Integer m_size;
  IRangeFunctor* m_functor;
  ParallelLoopOptions m_options;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OneTBBTaskImplementation::
~OneTBBTaskImplementation()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneTBBTaskImplementation::
initialize(Int32 nb_thread)
{
  if (nb_thread<0)
    nb_thread = 0;
  m_is_active = (nb_thread!=1);
  if (nb_thread!=0)
    m_p = new Impl(nb_thread);
  else
    m_p = new Impl();
  m_default_loop_options.setMaxThread(nbAllowedThread());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneTBBTaskImplementation::
terminate()
{
  m_p->terminate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 OneTBBTaskImplementation::
nbAllowedThread() const
{
  return m_p->nbAllowedThread();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneTBBTaskImplementation::
printInfos(ostream& o) const
{
  o << "OneTBBTaskImplementation"
    << " version=" << TBB_VERSION_STRING
    << " interface=" << TBB_INTERFACE_VERSION
    << " runtime_interface=" << TBB_runtime_interface_version();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneTBBTaskImplementation::
executeParallelFor(Integer begin,Integer size,const ParallelLoopOptions& options,IRangeFunctor* f)
{
  if (TaskFactory::verboseLevel()>=1)
    std::cout << "TBB: OneTBBTaskImplementation executeParallelFor begin=" << begin << " size=" << size << '\n';
  Integer max_thread = options.maxThread();
  // En exécution séquentielle, appelle directement la méthode \a f.
  if (max_thread==1 || max_thread==0){
    f->executeFunctor(begin,size);
    return;
  }

  // Remplace les valeurs non initialisées de \a options par celles de \a m_default_loop_options
  ParallelLoopOptions true_options(options);
  true_options.mergeUnsetValues(m_default_loop_options);

  ParallelForExecute pfe(this,true_options,begin,size,f);

  Integer nb_allowed_thread = m_p->nbAllowedThread();
  if (max_thread<0)
    max_thread = nb_allowed_thread;
  tbb::task_arena* used_arena = nullptr;
  if (max_thread<nb_allowed_thread)
    used_arena = m_p->m_sub_arena_list[max_thread];
  if (!used_arena)
    used_arena = &(m_p->m_main_arena);
  used_arena->execute(pfe);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneTBBTaskImplementation::
executeParallelFor(Integer begin,Integer size,Integer grain_size,IRangeFunctor* f)
{
  ParallelLoopOptions opt(m_default_loop_options);
  opt.setGrainSize(grain_size);
  executeParallelFor(begin,size,opt,f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OneTBBTaskImplementation::TaskThreadInfo* OneTBBTaskImplementation::
currentTaskThreadInfo() const
{
  Int32 thread_id = currentTaskThreadIndex();
  if (thread_id>=0)
    return m_p->threadTaskInfo(thread_id);
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 OneTBBTaskImplementation::
currentTaskIndex() const
{
  Int32 thread_id = currentTaskThreadIndex();
  if (thread_id<0 || thread_id>=m_p->nbAllowedThread())
    return 0;
  OneTBBTaskImplementation::TaskThreadInfo* tti = currentTaskThreadInfo();
  if (tti){
    Int32 task_index = tti->taskIndex();
    if (task_index>=0)
      return task_index;
  }
  return thread_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

ARCANE_REGISTER_APPLICATION_FACTORY(OneTBBTaskImplementation,ITaskImplementation,
                                    TBBTaskImplementation);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
