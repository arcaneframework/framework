// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TaskUnitTest.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Service de test des tâches.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/SpinLock.h"
#include "arcane/utils/Mutex.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/TestLogger.h"
#include "arccore/concurrency/internal/TaskFactoryInternal.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/Concurrency.h"
#include "arcane/core/ItemFunctor.h"
#include "arcane/core/ObserverPool.h"
#include "arcane/core/ItemPrinter.h"

#include "arcane/tests/TaskUnitTest_axl.h"

#include <thread>

long SerialFib(long n)
{
  if( n<2 )
    return n;
  else
    return SerialFib(n-1)+SerialFib(n-2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

namespace TaskTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Test creation de taches filles
class Test2
: public TraceAccessor
{
 public:

  explicit Test2(ITraceMng* tm)
  : TraceAccessor(tm)
  , m_wanted_nb_task(0)
  {}

 public:

  void exec()
  {
    m_nb = 0;
  
    ITask* master_task = TaskFactory::createTask(this,&Test2::_createSubTasks);
    m_wanted_nb_task = 100;
    master_task->launchAndWait();
    Int32 val = m_nb.load();
    info() << "** END_MY TEST2 n=" << val;
    if (val!=m_wanted_nb_task)
      ARCANE_FATAL("Bad value v={0} expected={1}",val,m_wanted_nb_task);
  }

  void _createSubTasks(const TaskContext& ctx)
  {
    //TaskFunctor<Test2> functor(this,&Test2::_testCallback);
    Int32 nb_task = m_wanted_nb_task;
    UniqueArray<ITask*> tasks(nb_task);
    ITask* parent_task = ctx.task();
    //ITask* master_task = TaskFactory::createTask(&functor);
    for( Integer i=0; i<nb_task; ++i ){
      ITask* t = TaskFactory::createChildTask(parent_task,this,&Test2::_testCallback);
      tasks[i] = t;
    }
    parent_task->launchAndWait(tasks);
  }

 private:

  void _testCallback()
  {
    ++m_nb;
  }
  std::atomic<Int32> m_nb = 0;
  Integer m_wanted_nb_task = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Test boucle sur les entites.
// Effectue un parallel_for et fait une reduction.
class Test3
: public TraceAccessor
{
 public:

  Test3(ITraceMng* tm,IMesh* mesh)
  : TraceAccessor(tm), m_mesh(mesh), m_node_coord(mesh->nodesCoordinates()),
    m_node_nb_access(VariableBuildInfo(mesh,"NodeNbAccess")),
    m_node_task_access(VariableBuildInfo(mesh,"NodeTaskAccess"))
  {
    m_total_value = 0.0;
    m_saved_value = 0.0;
    m_max_thread_index = (-1);
  }

 public:

  void exec()
  {
    m_total_value = 0.0;
    _testCallback(m_mesh->allNodes().view());
    m_saved_value = m_total_value;
    _exec1();
    _exec2();
    _exec3();
  }

  void _exec1()
  {
    {
      m_total_value = 0.0;
      arcaneParallelForeach(m_mesh->allNodes(), this, &Test3::_testCallback);
      _checkValid();
    }

    NodeVectorView nodes = m_mesh->allNodes().view();

    {
      m_total_value = 0.0;
      ParallelLoopOptions options;
      arcaneParallelForeach(nodes, options, this, &Test3::_testCallback);
      _checkValid();
    }

    {
      info() << "Test Static partitionner";
      m_total_value = 0.0;
      ParallelLoopOptions options;
      options.setPartitioner(ParallelLoopOptions::Partitioner::Static);
      arcaneParallelForeach(nodes, options, this, &Test3::_testCallback);
      _checkValid();
      info() << "End test Static partitionner";
    }

    for( Integer i=0; i<3; ++ i){
      Integer v = TaskFactory::verboseLevel();
      TaskFactory::setVerboseLevel(3);
      Integer grain_size = 10*i;
      _reset();
      info() << "Test Deterministic partitionner N=" << i << " grain_size=" << grain_size
             << " nb_item=" << nodes.size();
      m_total_value = 0.0;
      ParallelLoopOptions options;
      options.setGrainSize(grain_size);
      options.setPartitioner(ParallelLoopOptions::Partitioner::Deterministic);
      arcaneParallelForeach(nodes, options, this, &Test3::_testDeterministCallback);
      Integer nb_thread = TaskFactory::nbAllowedThread();
      _checkNbAccess(nb_thread);
      _checkValid();
      info() << "End test Deterministic partitionner";
      info() << "External loop: THREAD_IDX=" << TaskFactory::currentTaskThreadIndex()
             << " TASK_IDX=" << TaskFactory::currentTaskIndex();
      TaskFactory::setVerboseLevel(v);
    }
  }
  void _testDeterministCallback(NodeVectorView nodes)
  {
    Real local_total_coord = 0.0;
    Int32 thread_index = TaskFactory::currentTaskThreadIndex();
    Int32 task_index = TaskFactory::currentTaskIndex();
    if (nodes.size()>0)
      info() << "ITER N=" << nodes.size() << " FIRST_ID=" << nodes.localIds()[0] << " THREAD_IDX=" << thread_index
             << " TASK_IDX = " << task_index;
    ENUMERATE_NODE(inode,nodes){
      local_total_coord += m_node_coord[inode].squareNormL2();
      m_node_nb_access[inode] = m_node_nb_access[inode] + 1;
      m_node_task_access[inode] = task_index;
    }

    {
      SpinLock::ScopedLock s(m_reduce_lock);
      m_total_value += local_total_coord;
    }
  }
  void _reset()
  {
    m_total_value = 0.0;
    m_max_thread_index = (-1);
    ENUMERATE_NODE(inode,m_mesh->allNodes()){
      m_node_nb_access[inode] = 0;
      m_node_task_access[inode] = -1;
    }
  }
  void _exec2()
  {
    // Idem _exec1 mais avec lambda fonction
    m_total_value = 0.0;
    auto func = [this](NodeVectorView nodes)
    {
      Real local_total_coord = 0.0;
      Integer thread_index = TaskFactory::currentTaskThreadIndex();
      info() << "PARALLEL_LOOP size=" << nodes.size()
             << " thread_index=" << thread_index;
      ENUMERATE_NODE(inode,nodes){
        local_total_coord += m_node_coord[inode].squareNormL2();
      }

      {
        SpinLock::ScopedLock s(m_reduce_lock);
        m_total_value += local_total_coord;
        if (thread_index>m_max_thread_index)
          m_max_thread_index = thread_index;
      }
    };
    NodeVectorView nodes = m_mesh->allNodes().view();

    arcaneParallelForeach(nodes, func);
    _checkValid();

    // Teste avec options
    {
      info() << "Test ParallelLoopOptions";
      ParallelLoopOptions options;
      _reset();
      arcaneParallelForeach(nodes, options, func);
      _checkValid();
    }

    // Teste avec taille de bloc
    {
      info() << "Test ParallelLoopOptions block_size";
      ParallelLoopOptions options;
      options.setGrainSize(100);
      _reset();
      arcaneParallelForeach(nodes, options, func);
      _checkValid();
    }

    // Teste force sequentiel
    {
      info() << "Test ParallelLoopOptions force sequential";
      Integer nb_loop = 0;
      auto seq_func = [&](NodeVectorView nodes)
      {
        ++nb_loop;
        info() << "SEQUENTIAL LOOP";
        ENUMERATE_NODE(inode,nodes){
          m_total_value += m_node_coord[inode].squareNormL2();
        }
      };
      ParallelLoopOptions options;
      options.setMaxThread(1);
      _reset();
      arcaneParallelForeach(nodes, options, seq_func);
      if (nb_loop!=1){
        throw FatalErrorException(A_FUNCINFO,"Not a sequential execution");
      }
      _checkValid();
    }

    // Teste en changeant le nombre de threads.
    for( Integer x=0; x<32; ++x ){
      info() << "Test ParallelLoopOptions block_size=100 arena_size=" << x;
      ParallelLoopOptions options;
      options.setGrainSize(50);
      options.setMaxThread(x);
      _reset();
      arcaneParallelForeach(nodes, options, func);
      _checkValid();
      if (m_max_thread_index>x)
        ARCANE_FATAL("Bad max thread index v={0} max_expected={1}",m_max_thread_index,x);
    }
  }

  // Test pour le arcaneParallelForeach qui accepte un nombre arbitraire d'arguments
  void _exec3()
  {
    UniqueArray<Real> _x(m_mesh->allNodes().size());
    UniqueArray<Real> _y(m_mesh->allNodes().size());
    UniqueArray<Real> _z(m_mesh->allNodes().size());
    ENUMERATE_NODE(inode, m_mesh->allNodes()) {
      _x[inode->itemLocalId()] = m_node_coord[inode].x;
      _y[inode->itemLocalId()] = m_node_coord[inode].y;
      _z[inode->itemLocalId()] = m_node_coord[inode].z;
    }

    auto func = [this](ArrayView<Real> x, ArrayView<Real> y, ArrayView<Real> z)
    {
      Real local_total_coord = 0.0;
      Integer thread_index = TaskFactory::currentTaskThreadIndex();
      info() << "PARALLEL_LOOP size=" << x.size()
             << " thread_index=" << thread_index;
      for (auto i(0); i < x.size(); ++i) {
        Real3 _coord(x[i], y[i], z[i]);
        local_total_coord += _coord.squareNormL2();
      }

      {
        SpinLock::ScopedLock s(m_reduce_lock);
        m_total_value += local_total_coord;
        if (thread_index>m_max_thread_index)
          m_max_thread_index = thread_index;
      }
    };

    info() << "Test arcaneParallelForVa and multiples args";
    ParallelLoopOptions options;
    options.setGrainSize(100);
    ForLoopRunInfo fri(options);
    _reset();
    arcaneParallelForVa(fri, func, _x.view(), _y.view(), _z.view());
    _checkValid();
  }

  void _checkValid()
  {
    info() << "TOTAL_COORD=" << m_total_value;
    if (!math::isNearlyEqual(m_saved_value,m_total_value))
      ARCANE_FATAL("Bad parallel for total coords v={0} expected={1} diff={2}",
                   m_saved_value,m_total_value,(m_saved_value-m_total_value));
  }
  void _checkNbAccess(Integer nb_task)
  {
    Integer nb_error = 0;
    UniqueArray<Int32> nb_node_per_task(nb_task);
    nb_node_per_task.fill(0);
    ENUMERATE_NODE(inode,m_mesh->allNodes()){
      Integer n = m_node_nb_access[inode];
      if (n!=1){
        info() << "ERROR: bad nb_access for node=" << ItemPrinter(*inode) << " n=" << n;
        ++nb_error;
      }
      Integer task_access = m_node_task_access[inode];
      if (task_access<0 || task_access>=nb_task){
        info() << "ERROR: bad task_access for node=" << ItemPrinter(*inode) << " t=" << task_access;
        ++nb_error;
      }
      else{
        ++nb_node_per_task[task_access];
      }
    }
    for( Integer i=0; i<nb_task; ++i ){
      Integer n = nb_node_per_task[i];
      info() << "TASK_NB_NODE t=" << i << " n=" << n;
      // C'est une erreur si une tâche n'a pas traité de noeuds.
      if (n==0){
        info() << "ERROR: task '" << i << "' has no node";
        ++nb_error;
      }
    }
    if (nb_error!=0)
      ARCANE_FATAL("Bad nb access nb_error={0}",nb_error);
  }

 private:
  void _testCallback(NodeVectorView nodes)
  {
    Real local_total_coord = 0.0;
    ENUMERATE_NODE(inode,nodes){
      local_total_coord += m_node_coord[inode].squareNormL2();
    }

    {
      SpinLock::ScopedLock s(m_reduce_lock);
      m_total_value += local_total_coord;
    }
  }

  IMesh* m_mesh = nullptr;
  Real m_total_value = 0.0;
  Real m_saved_value = 0.0;
  Integer m_max_thread_index = 0;
  SpinLock m_reduce_lock;
  VariableNodeReal3 m_node_coord;
  VariableNodeInteger m_node_nb_access;
  VariableNodeInteger m_node_task_access;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Test parallel_for avec SpinLock et contention
class Test4
: public TraceAccessor
{
  class IAction
  {
   public:

    virtual ~IAction() = default;
    virtual Int32 value() =0;
    virtual void loop(Integer nb_loop) =0;
  };

  class DoAtomic : public IAction
  {
   public:
    DoAtomic() { m_nb = 0; }
    Int32 value() override { return m_nb.load(); }
    void loop(Integer nb_loop) override
    {
      for( int i=0; i<nb_loop; ++i )
        ++m_nb;
    }
    std::atomic<Int32> m_nb;
  };

  class DoSpinLock : public IAction
  {
   public:

    DoSpinLock() : m_nb(0) {}
    Int32 value() override { return m_nb; }
    void loop(Integer nb_loop) override
    {
      for( int i=0; i<nb_loop; ++i ){
        SpinLock::ScopedLock sl(m_lock);
        ++m_nb;
      }
    }
    Int32 m_nb;
    SpinLock m_lock;
  };

  class DoMutex
  : public IAction
  {
   public:
    DoMutex() : m_nb(0) {}
    Int32 value() override { return m_nb; }
    void loop(Integer nb_loop) override
    {
      for( int i=0; i<nb_loop; ++i ){
        Mutex::ScopedLock sl(m_lock);
        ++m_nb;
      }
    }
    Int32 m_nb = 0;
    Mutex m_lock;
  };

 public:

  explicit Test4(ITraceMng* tm)
  : TraceAccessor(tm)
  , m_action(nullptr)
  {}

 public:

  void exec()
  {
    int n = 100000;
    int n2 = 1000;
    ParallelLoopOptions loop_options;
    loop_options.setGrainSize(1000);
    {
      DoAtomic atomic_action;
      m_action = &atomic_action;
      Real v1 = platform::getRealTime();
      ParallelLoopOptions loop_options2;
      loop_options2.setGrainSize(n2);
      arcaneParallelFor(0, n, loop_options2, [&](Integer a, Integer n) { _doLoop(a, n); });
      Real v2 = platform::getRealTime();
      _print("atomic",m_action->value(),v2-v1,n);
    }
    {
      DoSpinLock spinlock_action;
      m_action = &spinlock_action;
      Real v1 = platform::getRealTime();
      arcaneParallelFor(0, n, loop_options, [&](Integer a, Integer n) { _doLoop(a, n); });
      Real v2 = platform::getRealTime();
      _print("spin", m_action->value(), v2 - v1, n);
    }
    {
      DoMutex mutex_action;
      m_action = &mutex_action;
      Real v1 = platform::getRealTime();
      arcaneParallelFor(0, n, loop_options, [&](Integer a, Integer n) { _doLoop(a, n); });
      Real v2 = platform::getRealTime();
      _print("mutex", m_action->value(), v2 - v1, n);
    }
    {
      // Test Mutex avec syntaxe des lambda fonction du C++0x
      DoMutex mutex_action;
      m_action = &mutex_action;
      Real v1 = platform::getRealTime();
      arcaneParallelFor(0, n, loop_options, [this](Integer /*i0*/, Integer size) {
        m_action->loop(size * 10);
      });
      Real v2 = platform::getRealTime();
      _print("mutex_c++0x", m_action->value(), v2 - v1, n);
    }
  }

  void _print(const String& name,Int32 value,Real elapsed_time,Integer n)
  {
    info() << "TEST V=" << value << " " << name
           << " time=" << elapsed_time << " time2=" << elapsed_time / (n*10);
  }

  void _doLoop(Integer /*i0*/, Integer size)
  {
    m_action->loop(size*10);
  }

 private:

  IAction* m_action = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Teste création de tâches imbriquées.
 * Utilise exemple des tbb sur la suite de fibonnaci.
 */
class Test5Fibonnaci
{
 public:

  const long n = 0;
  long* const sum = nullptr;

  Test5Fibonnaci( long n_, long* sum_ ) : n(n_), sum(sum_)
  {}

  void execute(const TaskContext& context);

  static long SerialFib( long n ) {
    if( n<2 )
      return n;
    else
      return SerialFib(n-1)+SerialFib(n-2);
  }

  static long ParallelFib( long n )
  {
    long sum = 0;
    Test5Fibonnaci a(n,&sum);
    ITask* task = TaskFactory::createTask(&a,&Test5Fibonnaci::execute);
    task->launchAndWait();
    return sum;
  }
 public:

  static std::atomic<Int32> m_nb_exec;
};
std::atomic<Int32> Test5Fibonnaci::m_nb_exec(0);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Test5Fibonnaci::
execute(const TaskContext& context)
{
  // Pour compter le nombre de tâches mais attention
  // cela ralentit considérablement l'exécution sur les
  // noeuds multi-processeurs.

  // ++m_nb_exec;

  if( n<10 ) {
    *sum = SerialFib(n);
    return;
  }

  long x = 0;
  long y = 0;
  Test5Fibonnaci a(n-1,&x);
  Test5Fibonnaci b(n-2,&y);
  ITask* child_tasks[2];
  ITask* parent_task = context.task();
  child_tasks[0] = TaskFactory::createChildTask(parent_task,&a,&Test5Fibonnaci::execute);
  child_tasks[1] = TaskFactory::createChildTask(parent_task,&b,&Test5Fibonnaci::execute);
  parent_task->launchAndWait(ConstArrayView<ITask*>(2,child_tasks));
  
  // Effectue la somme
  *sum = x+y;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Test l'utilisation du contexte.
 */
class Test6
: public TraceAccessor
{
 public:

  Test6(ITraceMng* tm,Int32 first_value,Int32 nb_value,Int32 step_size=100)
  : TraceAccessor(tm), m_first_value(first_value), m_nb_value(nb_value),
    m_step_size(step_size)
  {
  }

  void exec()
  {
    // Implémentation obsolète
    m_value = 0;
    Int64 n0 = m_first_value-1;
    Int32 nb = m_nb_value;
    Int64 n1 = n0 + nb;
    info() << "T6_Exec first=" << m_first_value << " n=" << m_nb_value;
    ParallelLoopOptions loop_options;
    loop_options.setGrainSize(m_step_size);
    arcaneParallelFor(m_first_value,nb,loop_options,[&](Integer a,Integer n){ loop(a,n); });
    // \a m_value doit être égal à la somme des n1 premiers entiers
    // moins la somme des n0 premiers entiers.
    Int64 n0_sum = (n0 * (n0 + 1)) / 2;
    Int64 n1_sum = (n1 * (n1 + 1)) / 2;
    Int64 expected = n1_sum - n0_sum;
    if (m_value!=expected)
      ARCANE_FATAL("Bad sum expected={0} value={1}",expected,m_value);
  }

  void loop(Integer begin,Integer size)
  {
    info() << "TEST6-HI begin=" << begin << " size=" << size
           << " thread_index=" << TaskFactory::currentTaskThreadIndex()
           << " thread_id=" << std::this_thread::get_id()
    ;
    for( Integer i=begin; i<(begin+size); ++i ){
      SpinLock::ScopedLock sl(m_lock);
      m_value += i;
    }
  }
  Int64 m_value = 0;
  Int32 m_first_value = 0;
  Int32 m_nb_value = 0;
  Int32 m_step_size = 0;
  SpinLock m_lock;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace TaskTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test des taches
 */
class TaskUnitTest
: public ArcaneTaskUnitTestObject
{
 public:

  explicit TaskUnitTest(const ServiceBuildInfo& cb);
  ~TaskUnitTest() override;

 public:

  void initializeTest() override;
  void executeTest() override;

 private:
  
  void _createTheadCallback()
  {
    info() << "OBSERVER THREAD CALLBACK !";
  }

 private:

  ObserverT<TaskUnitTest> m_thread_create_observer;
  bool m_has_thread_callback = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_TASKUNITTEST(TaskUnitTest,TaskUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TaskUnitTest::
TaskUnitTest(const ServiceBuildInfo& mb)
: ArcaneTaskUnitTestObject(mb)
, m_thread_create_observer(this, &TaskUnitTest::_createTheadCallback)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TaskUnitTest::
~TaskUnitTest()
{
  if (m_has_thread_callback)
    TaskFactoryInternal::removeThreadCreateObserver(&m_thread_create_observer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TaskUnitTest::
executeTest()
{
  ValueChecker vc(A_FUNCINFO);
  {
    TaskTest::Test2 t2(traceMng());
    t2.exec();
  }
  {
    TaskTest::Test3 t3(traceMng(),mesh());
    t3.exec();
  }
  {
    TaskTest::Test4 t4(traceMng());
    t4.exec();
  }

  Integer nb_loop = 2;
  if (1){
    TaskTest::Test5Fibonnaci::m_nb_exec = 0;
    Real v1 = platform::getRealTime();
    for( Integer i=0; i<nb_loop; ++i ){
      long z1 = TaskTest::Test5Fibonnaci::ParallelFib(25);
      info() << "F[25] =" << z1;
      vc.areEqual(z1,75025L,"F[25]");
      long z2 = TaskTest::Test5Fibonnaci::ParallelFib(35);
      info() << "F[35] =" << z2;
      vc.areEqual(z2,9227465L,"F[35]");
      long z3 = TaskTest::Test5Fibonnaci::ParallelFib(40);
      info() << "F[40] =" << z3;
      vc.areEqual(z3,102334155L,"F[45]");
      info() << "NB_EXEC=" << TaskTest::Test5Fibonnaci::m_nb_exec.load();
    }
    Real v2 = platform::getRealTime();
    info() << "TIME=" << (v2-v1);
  }

  { TaskTest::Test6 t6(traceMng(),50,1000); t6.exec(); }
  { TaskTest::Test6 t6(traceMng(),0,1500); t6.exec(); }
  { TaskTest::Test6 t6(traceMng(),0,32000); t6.exec(); }
  { TaskTest::Test6 t6(traceMng(),0,32000,500); t6.exec(); }
  { TaskTest::Test6 t6(traceMng(),127,1329); t6.exec(); }
  { TaskTest::Test6 t6(traceMng(),47,721,0); t6.exec(); }
  { TaskTest::Test6 t6(traceMng(),1023,4097,50); t6.exec(); }
  { TaskTest::Test6 t6(traceMng(),0,4000,100); t6.exec(); }
  { TaskTest::Test6 t6(traceMng(),0,200000,2000); t6.exec(); }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Func
{
 public:
  void exec(Integer begin,Integer size)
  {
    cout << "HI " << begin << " " << size << '\n';
  }
};

void TaskUnitTest::
initializeTest()
{
  ParallelLoopOptions default_options = TaskFactory::defaultParallelLoopOptions();
  info() << "InitializeTest: DefaultOptions GrainSize=" << default_options.grainSize()
         << " partitioner=" << (int)default_options.partitioner();

  TestLogger::stream() << "DefaultParallelLoopGrainSize=" << default_options.grainSize() << "\n";
  TestLogger::stream() << "DefaultParallelLoopPartitioner=" << (int)default_options.partitioner() << "\n";

  info() << "InitializeTest: THREAD_IDX=" << TaskFactory::currentTaskThreadIndex()
         << " TASK_IDX=" << TaskFactory::currentTaskIndex();

  TaskFactory::setVerboseLevel(1);
  // Cette boucle doit être la première pour tester l'observable sur
  // la création de threads.
  TaskFactoryInternal::addThreadCreateObserver(&m_thread_create_observer);
  m_has_thread_callback = true;

  Func my_functor;
  ParallelLoopOptions loop_options;
  loop_options.setGrainSize(100);
  arcaneParallelFor(50,1000,loop_options,[&](Integer a,Integer n){ my_functor.exec(a,n); });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
