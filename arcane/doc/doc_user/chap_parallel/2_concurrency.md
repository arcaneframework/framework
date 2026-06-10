# Concurrency and Multi-threading {#arcanedoc_parallel_concurrency}

[TOC]

<!-- describes the notion of concurrency and parallelization at the loop level -->

The notion of concurrency is implemented in %Arcane via the notion of a task.

This task notion allows the concurrent execution of multiple operations via
threads.

This notion is complementary to the notion of domain decomposition used by
Arcane::IParallelMng. It is therefore entirely possible to mix domain
decomposition and threads.

\warning Nevertheless, if the implementation of Arcane::IParallelMng is done via
MPI, it is not recommended to call Arcane::IParallelMng when tasks are running
concurrently, for example in parallelized loops. Most MPI implementations are
not very performant in this mode, and some only support it partially.

To use tasks, you must include the following file:

```cpp
#include "arcane/Concurrency.h"
```

There are two mechanisms for using tasks:

1. Implicitly via the notion of a parallel loop
2. Explicitly by creating tasks directly

The first solution is the simplest and should be considered first.

## Activation {#arcanedoc_parallel_concurrency_activation}

By default, concurrency support is disabled. Activation is done **before**
launching the code, by specifying the number of tasks that can run concurrently
on the command line (see page \ref arcanedoc_execution_launcher to find out how
to do this).

It is possible to check in the code whether concurrency is active by calling the
Arcane::TaskFactory::isActive() method.

It is not possible to activate concurrency during execution.

## Parallel Loops {#arcanedoc_parallel_concurrency_parallel_for}

There are two forms of parallel loops. The first form applies to classic loops,
the second to groups of entities.

The operating mechanism is similar to the `omp parallel for` directives in
OpenMp.

\warning The user of this mechanism must ensure that the loop can be correctly
parallelized without edge effects. Specifically, this includes (but is not
limited to) the guarantee that the loop iterations are independent, and that
there are no loop exit operations (return, break).

The first form is for parallelizing the following sequential loop:

```cpp
void func()
{
  for( Integer i=0; i<n; ++i )
    p[i] = (gamma[i]-1) * rho[i] * e[i];
}
```

Parallelization is done as follows: you must first write a functor class that
represents the operation you wish to perform over an iteration interval. Then,
you must use the arcaneParallelFor() operation, specifying this functor as an
argument, as follows:

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

This syntax is a bit verbose. If the compiler supports the C++11 standard, it is
possible to use lambda functions to simplify the writing:

```cpp
void func()
{
  Arcane::arcaneParallelFor(0,n,[&](Integer begin,Integer size){
     for( Integer i=begin; i<(begin+size); ++i )
       p[i] = (gamma[i]-1.0) * rho[i] * e[i];
  });
}
```

A specialization exists for groups of entities. To parallelize an enumeration
over a group like the following code:

```cpp
void func()
{
  ENUMERATE_CELL(icell,my_group){
    p[icell] = (gamma[icell]-1.0) * rho[icell] * e[icell];
  }
}
```

You must write it like this:

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

Similarly, with C++11 support, you can simplify:

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

For the Arcane::arcaneParallelFor() and Arcane::arcaneParallelForeach() loops,
it is possible to pass an instance of ParallelLoopOptions as an argument to
configure the parallel loop. For example, it is possible to specify the interval
size to divide the loop:

```cpp
void func()
{
  Arcane::ParallelLoopOptions options;
  // Executes the loop in chunks of about 50 cells.
  options.setGrainSize(50);
  Arcane::arcaneParallelForeach(my_group,options,[&](Arcane::CellVectorView cells){
    ENUMERATE_CELL(icell,cells){
      p[icell] = (gamma[icell]-1.0) * rho[icell] * e[icell];
    }
  });
}
```

## Explicit Task Usage {#arcanedoc_parallel_concurrency_task}

Creating a task is done via the task factory. You must specify a functor as an
argument in the same way as with parallel loops:

```cpp
class Func
{
  public:
   void exec(const TaskContext& ctx)
   {
     // Execute the task.
   }
};

void func()
{
  Func my_functor
  Arcane::ITask* master_task = Arcane::TaskFactory::createTask(&my_functor,&Func::exec);
}
```

Once the task is created, it is possible to launch it and wait for its
termination using the ITask::launchAndWait() method. For simplicity reasons, the
task is not launched until this method has been called.

It is possible to create sub-tasks from a primary task using the
Arcane::TaskFactory::createChildTask() method. The user must manage the
launching and waiting of sub-tasks. For example:

```cpp
using namespace Arcane;
ITask* master_task = TaskFactory::createTask(...);
UniqueArray<ITask*> sub_tasks;
sub_tasks.add(TaskFactory::createChildTask(master_task,&my_functor,&Func::exec);
sub_tasks.add(TaskFactory::createChildTask(master_task,&my_functor,&Func::exec);
master_task->launchAndWait(sub_tasks);
```

The following complete example shows the implementation of calculating a
Fibonacci sequence using the task mechanism.

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

     // Perform the sum
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
