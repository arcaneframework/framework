# Utilization of Accelerators (GPU) {#arcanedoc_parallel_accelerator}

<!-- briefly presents the use of accelerators in %Arcane. -->

[TOC]

In this chapter, we will call an accelerator a dedicated co-processor
different from the main processor used to execute the calculation code. In the
current version of %Arcane, these are GPGPU type accelerators.

The %Arcane API for managing accelerators is inspired by libraries
such as [RAJA](https://github.com/LLNL/RAJA) or
[Kokkos](https://github.com/kokkos/kokkos) but is restricted to the specific
needs of %Arcane.

\note The %Arcane accelerator API can be used independently of
the mechanisms associated with calculation codes such as modules, mesh, or
services. For an example of standalone operation, refer to chapter
\ref arcanedoc_parallel_accelerator_standalone.

The current implementation only supports NVIDIA graphics cards (via CUDA) or
AMD (via ROCm) as accelerators.

The %Arcane accelerator API meets the following objectives:

- unify the behavior between sequential CPU, multi-threaded CPU, and
  accelerator.
- have a single executable and be able to dynamically choose where the code will
  be executed: CPU or accelerator (or both at once).
- have source code independent of the compiler, so we do not use mechanisms such
  as `#pragma` as in OpenMP or OpenACC standards.

\note If you wish to use %Arcane on both GPU and CPU for the CUDA environment,
it is strongly recommended to use `clang` as the compiler instead of `nvcc`
because the latter generates less performant code on the CPU side. This is due
to the use of `std::function` to encapsulate the lambdas used in %Arcane (see
[New Compiler Features in CUDA 8](https://developer.nvidia.com/blog/new-compiler-features-cuda-8/#extended___host_____device___lambdas)
for more information)

The operating principle is the execution of offloaded compute kernels. The code
is executed by default on the CPU (the host) and certain parts of the
calculation are offloaded to the accelerators. This offloading is done via
specific calls.

To use the accelerators, it is necessary to have compiled %Arcane with CUDA or
ROCm. More information is in chapter \ref arcanedoc_build_install_build.

## Usage in Arcane {#arcanedoc_parallel_accelerator_usage}

All types used for accelerator management are in the Arcane::Accelerator
namespace. There are two components for managing accelerators:

- `arcane_accelerator_core` whose header files are included via
  `#include <arcane/accelerator/core>`. This component contains classes
  independent of the accelerator type.

- `arcane_accelerator` whose header files are included via
  `#include <arcane/accelerator>`. This component contains classes that allow
  offloading compute kernels to a specific accelerator.

The main classes for managing accelerators are:

- \arcaneacc{IAcceleratorMng} which allows access to the default execution
  environment.
- \arcaneacc{Runner} which represents an execution environment
- \arcaneacc{RunQueue} which represents an execution queue
- \arcaneacc{RunCommand} which represents a command (a compute kernel)
  associated with an execution queue.

There are two ways to use accelerators in %Arcane:

- via an instance of \arcaneacc{IAcceleratorMng} created and initialized by
  %Arcane when the executable is launched
  (\ref arcanedoc_parallel_accelerator_module). This is the recommended method.
- via an instance of \arcaneacc{Runner} created and manually initialized
  (\ref arcanedoc_parallel_accelerator_runner).

To run a calculation on an accelerator, you must instantiate an execution queue.
The \arcaneacc{RunQueue} class manages such a queue. The \arcaneacc{makeQueue()}
function allows creating such a queue. Execution queues can be temporary or
persistent but cannot be copied. The \arcaneacc{makeQueueRef()} method allows
creating a reference to a queue that can be copied.

\note By default, creating a \arcaneacc{RunQueue} from an \arcaneacc{Runner} is
not thread-safe for performance reasons. If you want to be able to launch
multiple execution queues from the same \arcaneacc{Runner} instance, you must
call the method \arcaneacc{Runner::setConcurrentQueueCreation(true)} beforehand

### Usage in modules {#arcanedoc_parallel_accelerator_module}

Any module can retrieve an implementation of the \arcaneacc{IAcceleratorMng}
interface via the method \arcane{AbstractModule::acceleratorMng()}. The
following code example shows how to use accelerators from an entry point:

```cpp
// File to include all the time
#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/RunQueue.h"

// File to include to have RUNCOMMAND_ENUMERATE
#include "arcane/accelerator/RunCommandEnumerate.h"

// File to include to have RUNCOMMAND_LOOP
#include "arcane/accelerator/RunCommandLoop.h"

using namespace Arcane;
using namespace Arcane::Accelerator;

class MyModule
: public Arcane::BasicModule
{
 public:
  void myEntryPoint()
  {
    RunQueue* queue = acceleratorMng()->defaultQueue();
    // Loop over cells offloaded to accelerator
    auto command1 = makeCommand(queue);
    command1 << RUNCOMMAND_ENUMERATE(Cell,vi,allCells()){
    };
  
    // Classic 1D loop offloaded to accelerator
    auto command2 = makeCommand(queue)
    command2 << RUNCOMMAND_LOOP1(iter,5){
    };
  }
};
```

### Specific Runner Instance {#arcanedoc_parallel_accelerator_runner}

It is possible to create multiple instances of the \arcaneacc{Runner} object.

An instance of this class is associated with an execution policy whose possible
values are given by the enumeration \arcaneacc{eExecutionPolicy}. By default,
the execution policy is \arcaneacc{eExecutionPolicy::Sequential}, which means
that the compute kernels will be executed sequentially.

\note When creating an instance of \arcaneacc{Runner} on an accelerator, it is
possible to specify an accelerator other than the default accelerator (if
multiple are available). This significantly complicates memory management.
Chapter \ref arcanedoc_parallel_accelerator_multi explains how to handle this.

It is also possible to automatically initialize an instance of this class based
on command-line arguments:

```cpp
#include "arcane/accelerator/RunQueue.h"
using namespace Arcane;
using namespace Arcane::Accelerator;
Runner runner;
ITraceMng* tm = ...;
IApplication* app = ...;
initializeRunner(runner,tm,app->acceleratorRuntimeInitialisationInfo());
```

## Compilation {#arcanedoc_parallel_accelerator_compilation}

%Arcane provides integration to compile with accelerator support via CMake.
Those who use another build system must manage this support similarly.

To be able to use compute kernels on an accelerator, you generally need to use a
specific compiler. For example, the current implementation of %Arcane via CUDA
uses NVIDIA's `nvcc` compiler for this. This compiler is responsible for
compiling the part associated with the accelerator. The part associated with the
CPU is compiled with the same compiler as the rest of the code.

It is necessary to specify in the `CMakeLists.txt` that you want to use
accelerators as well as the files that will be compiled for the accelerators.
Only files using commands (RUNCOMMAND_LOOP or RUNCOMMAND_ENUMERATE) need to be
compiled for the accelerators. For this, %Arcane defines the following CMake
functions:

- **arcane_accelerator_enable()** which must be called before other functions to
  detect the compiler environment for the accelerator
- **arcane_accelerator_add_source_files(file1.cc [file2.cc] ...)** to indicate
  the source files that must be compiled on accelerators
- **arcane_accelerator_add_to_target(mytarget)** to indicate that the target
  `mytarget` requires the accelerator environment.

If %Arcane is compiled in a CUDA environment, the CMake variable
`ARCANE_HAS_CUDA` is defined. If %Arcane is compiled in a HIP/ROCm environment,
then `ARCANE_HAS_HIP` is defined.

## Execution {#arcanedoc_parallel_accelerator_exec}

The choice of the default execution environment
(\arcaneacc{IAcceleratorMng::defaultRunner()}) is determined by the command
line:

- If the `AcceleratorRuntime` option is specified, that runtime is used.
  Currently, the only possible values are `cuda` or `hip`. For example:
  ```sh
  MyExec -A,AcceleratorRuntime=cuda data.arc
  ```
- Otherwise, if multi-threading is enabled via the `-T` option (see
  \ref arcanedoc_execution_launcher), then the compute kernels are distributed
  across multiple threads,
- Otherwise, the compute kernels are executed sequentially.

## Compute Kernels (RunCommand) {#arcanedoc_parallel_accelerator_runcommand}

Once you have an instance of \arcaneacc{RunQueue}, it is possible to create a
command that can be offloaded to the accelerator. Commands are always loops that
can take the following forms:

- Classic loop from dimension 1 to 4. This is done via the macros
  RUNCOMMAND_LOOP(), RUNCOMMAND_LOOP1(), RUNCOMMAND_LOOP2(), RUNCOMMAND_LOOP3()
  or RUNCOMMAND_LOOP4().
- Loop over mesh entities. This is done via the RUNCOMMAND_ENUMERATE() macro.

Chapter \ref arcanedoc_parallel_accelerator_lambda describes the syntax of these
loops.

The following code example shows how to use accelerators from an entry point:

```cpp
// Files to include all the time
#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/RunQueue.h"

// File to include to have RUNCOMMAND_ENUMERATE
#include "arcane/accelerator/RunCommandEnumerate.h"

// File to include to have RUNCOMMAND_LOOP
#include "arcane/accelerator/RunCommandLoop.h"

using namespace Arcane;
using namespace Arcane::Accelerator;

class MyModule
: public Arcane::BasicModule
{
 public:
  void myEntryPoint()
  {
    RunQueue queue = ...;

    // Loop over cells offloaded to accelerator
    auto command1 = makeCommand(queue);
    command1 << RUNCOMMAND_ENUMERATE(Cell,vi,allCells()){
    };

    // Classic 1D loop offloaded to accelerator
    auto command2 = makeCommand(queue)
    command2 << RUNCOMMAND_LOOP1(iter,5){
    };
  }
};
```

### Usage of Views {#arcanedoc_parallel_accelerator_view}

Accelerators generally have their own memory, which is different from the host's
memory. It is therefore necessary to specify how the data will be used to manage
potential transfers between memories. For this, %Arcane provides a mechanism
called a view, which allows specifying for a variable or an array whether it
will be used as input, output, or both.

\warning A view is a **TEMPORARY** object and is always associated with a
command (\arcaneacc{RunCommand}) and a container (Arcane Variable or array) and
must not be used when the associated command is finished or the associated
container is modified.

%Arcane offers views on variables (\arcane{VariableRef}) or on the
\arcane{NumArray} class (The page \ref arcanedoc_core_types_numarray describes
the use of this class in more detail).

Regardless of the associated container, the declaration of views is the same and
uses the methods \arcaneacc{viewIn()}, \arcaneacc{viewOut()} or
\arcaneacc{viewInOut()}.

```cpp
// To have NumArray
#include "arcane/utils/NumArray.h"

// To have views on variables
#include "arcane/accelerator/VariableViews.h"

// To have views on NumArray
#include "arcane/accelerator/NumArrayViews.h"

Arcane::Accelerator::RunCommand& command = ...;
// 1D arrays
Arcane::NumArray<Real,MDDim1> a;
Arcane::NumArray<Real,MDDim1> b;
Arcane::NumArray<Real,MDDim1> c;

// 1D variable on cells
VariableCellReal var_c = ...;

// Input view (read-only)
auto in_a = viewIn(command,a);

// Input/output view
auto inout_b = viewInOut(command,b);

// Output view (write-only) on the variable 'var_c'
auto out_c = viewOut(command,var_c);
```

### Memory Management of Data Managed by Arcane

By default, %Arcane uses the allocator returned by
\arcane{MeshUtils::getDefaultDataAllocator()} for the \arcane{NumArray} type as
well as all variables (\arcane{VariableRef}), entity groups (\arcane{ItemGroup})
and connectivities.

When using accelerators, %Arcane requires that this allocator allocates memory
that is accessible both on the host and the accelerator. This means that the
data corresponding to these objects is accessible both on the host (CPU) and on
the accelerators. For this, %Arcane uses unified memory
(\arccore{eMemoryResource::UnifiedMemory}) by default.

With unified memory, the accelerator automatically manages potential memory
transfers between the accelerator and the host. These transfers can be
time-consuming if they are frequent, but if a piece of data is only used
on the CPU or on the accelerator, there will be no memory transfers and thus
performance will not be impacted.

Starting from version 3.14.12 of %Arcane, it is possible to change the default
memory resource used via the environment variable
`ARCANE_DEFAULT_DATA_MEMORY_RESOURCE`. On accelerators where the memory
\arccore{eMemoryResource::Device} is directly accessible from the host (for
example MI250X, MI300A, GH200), this allows avoiding transfers that unified
memory might cause.

In all cases, it is possible to specify a specific allocator for
\arccore{UniqueArray} and \arcane{NumArray} via the methods
\arcane{MemoryUtils::getAllocator()} or
\arcane{MemoryUtils::getAllocationOptions()}.

%Arcane provides mechanisms for providing information to optimize this memory
management. These mechanisms depend on the accelerator type and may not be
available everywhere. They are accessible via the method
\arcaneacc{Runner::setMemoryAdvice()}.

Starting from version 3.10 of %Arcane and with NVIDIA accelerators, %Arcane
offers features to detect memory transfers between the CPU and the accelerator.
The page \ref arcanedoc_debug_perf_cupti describes this functionality.

### Example of using a complex loop {#arcanedoc_parallel_accelerator_complexloop}

The following example shows how to modify the iteration range so that it does
not start from zero:

```cpp
using namespace Arcane;
using namespace Arcane::Accelerator;
{
  Arcane::Accelerator::Runner runner = ...;
  auto queue = makeQueue(runner);
  auto command = makeCommand(queue);
  auto out_t1 = viewOut(command,t1);
  Int64 base = 300;
  Int64 s1 = 400;
  auto b = makeLoopRanges({base,s1},n2,n3,n4);
  command << RUNCOMMAND_LOOP(iter,b)
  {
    auto [i, j, k, l] = iter();
    out_t1(i,j,k,l) = _getValue(i,j,k,l);
  };
}
```

### Using lambdas {#arcanedoc_parallel_accelerator_lambda}

Regardless of the macro (RUNCOMMAND_ENUMERATE(), RUNCOMMAND_LOOP(), ...) used
for the loop, the following code must be
a [C++11 lambda function](https://en.cppreference.com/w/cpp/language/lambda). It
is this lambda function that will eventually be offloaded to the accelerator.

%Arcane uses the `operator<<` to "send" the loop to a command
(\arcaneacc{RunCommand}), which allows writing the code similarly to a classic
C++ loop (or an ENUMERATE_() loop in the case of mesh entities) with the
following few modifications:

- curly braces (`{` and `}`) are mandatory
- a `;` must be added after the last brace.
- the body of a lambda is a function, not a loop. Consequently, it is not
  possible to use keywords such as `continue` or `break`. The keyword `return`
  is available and therefore will have the same effect as `continue` in a loop.

For example:

```cpp
Arcane::Accelerator::RunCommand& command = ...
// 1D loop of 'nb_value' with 'iter' the iterator
command << RUNCOMMAND_LOOP1(iter,nb_value)
{
  // Code executed on accelerator
};
```

```cpp
Arcane::Accelerator::RunCommand& command = ...
// Loop over the cells of the group 'my_group' with 'cid' the index of
// the current cell (of type Arcane::CellLocalId)
command << RUNCOMMAND_ENUMERATE(Cell,icell,my_group)
{
  // Code executed on accelerator
};
```

When a computation kernel is offloaded to the accelerator, you must not access
the memory associated with the views from another part of the code during
execution, or it may crash. Generally, this can only happen when the
\arcaneacc{RunQueue} are asynchronous. For example:

```cpp
#include "arcane/accelerator/Views.h"
using namespace Arcane::Accelerator;
Arcane::Accelerator::RunQueue& queue = ...;
queue.setAsync(true);
Arcane::NumArray<Real,MDDim1> a;
Arcane::NumArray<Real,MDDim1> b;

Arcane::Accelerator::RunCommand& command = makeCommand(queue);
auto in_a = viewIn(command,a);
auto out_b = viewOut(command,b);
// Copy A into B
command << RUNCOMMAND_LOOP1(iter,nb_value)
{
  auto [i] = iter();
  out_b(i) = in_a(i);
};
// The command is running as long as the barrier() method
// has not been called

// HERE you MUST NOT use 'a' or 'b' or 'in_a' or 'out_b'

queue.barrier();

// HERE you can use 'a' or 'b' (BUT NOT 'in_a' or 'out_b' because the
// command is finished)
```

### Limitation of C++ lambdas on accelerators {#arcanedoc_parallel_accelerator_limitlambda}

The compilation mechanisms and memory management on accelerators impose
restrictions on the use of classic C++ lambdas

#### Calling other functions in lambdas {#arcanedoc_parallel_accelerator_callslambda}

In a lambda intended to be offloaded to the accelerator, you can only call:

- class methods that are **public**
- functions that are `inline`
- functions or methods that have the ARCCORE_HOST_DEVICE or ARCCORE_DEVICE
  attribute or `constexpr` methods

It is not possible to call external functions defined in other compilation
units (for example, other libraries)

#### Using fields of a class instance {#arcanedoc_parallel_accelerator_classinstance}

You must not use a reference to a class field in lambdas because it is captured
by reference. This will cause a crash due to invalid memory access on the
accelerator. To avoid this problem, simply declare a local copy of the class
instance value you wish to use within the function. In the following example,
the function `f1()` will cause a crash while `f2()` will work correctly.

```cpp
class A
{
 public:
  void f1();
  void f2();
  int my_value;
};
void A::f1()
{
  Arcane::Accelerator::RunCommand& command = ...
  Arcane::NumArray<int,MDDim1> a(100);
  auto out_a = viewIn(command,a);
  command << RUNCOMMAND_LOOP1(iter,100){
    out_a(iter) = my_value+5; // BAD !!
  };
}
void A::f2()
{
  Arcane::Accelerator::RunCommand& command = ...
  Arcane::NumArray<int,MDDim1> a(100);
  auto out_a = viewIn(command,a);
  int v = my_value;
  command << RUNCOMMAND_LOOP1(iter,100){
    out_a(iter) = v+5; // GOOD !!
  };
}
```

## Using the message exchange mechanism

Starting from version 3.10, %Arcane supports "Accelerator Aware" MPI libraries.
In this case, the buffer used for variable synchronizations is allocated
directly on the accelerator. If a variable is used on the accelerator, this
avoids unnecessary copies between the host and the accelerator. Shared memory
message exchange mode also supports this mechanism.

If problems occur, this support can be disabled by setting the environment
variable `ARCANE_DISABLE_ACCELERATOR_AWARE_MESSAGE_PASSING` to a non-zero value.

## Multi-accelerator Management {#arcanedoc_parallel_accelerator_multi}

%Arcane associates an instance of \arcaneacc{Runner} (accessible via
\arcane{ISubDomain::acceleratorMng()}) when creating a subdomain.
When a machine has multiple accelerators, %Arcane by default chooses the first
one returned in the available accelerators. This behavior can be changed by
setting the environment variable
`ARCANE_ACCELERATOR_PARALLELMNG_RANK_FOR_DEVICE` to a strictly positive value
indicating the modulo between the subdomain rank (returned by
\arcane{IParallelMng::commRank()} of \arcane{ISubDomain::parallelMng()}) and the
accelerator index in the list of accelerators. For example, if this environment
variable is 8, then the subdomain of rank N will be associated with the
accelerator of index \a (N % 8). For this mechanism to work, the value of this
environment variable must therefore be less than the number of accelerators
available on the machine.

### Memory Management

When multiple accelerators are available on the same machine, there is generally
a "current" accelerator for each thread (for example, with CUDA it is possible
to retrieve it using the `cudaGetDevice()` method and change it using the
`cudaSetDevice()` method). When allocating memory on the accelerator, it is done
on this "current" accelerator, and this memory will not be available on other
accelerators. An instance of \arcaneacc{RunQueue} is associated with a given
accelerator, so you must ensure that the memory regions used by a command are
accessible. If this is not the case, it will produce an error during execution
(For example, with CUDA, this is error 400, whose message is "invalid resource
handle").

If the "current" accelerator has been changed, for example, when calling an
external library, it is possible to change it by calling the method
\arcaneacc{Runner::setAsCurrentDevice()}.

## Managing Connectivity and Entity Information

Accessing mesh connectivity is done differently on the accelerator than on the
CPU for performance reasons. Specifically, it is not possible to use classic
entities (\arcane{Cell},\arcane{Node}, ...). Instead, you must use local
identifiers such as \arcane{CellLocalId} or \arcane{NodeLocalId}.

The \arcane{UnstructuredMeshConnectivityView} class allows access to
connectivity information. It is possible to define an instance of this class and
keep it during the calculation. To initialize the instance, you must call the
method \arcane{UnstructuredMeshConnectivityView::setMesh()}.

\warning Like all views, the instance is invalidated when the mesh changes.
Therefore, you must call \arcane{UnstructuredMeshConnectivityView::setMesh()}
again after modifying the mesh.

To access generic entity information, such as type or owner, you must use the
\arcane{ItemGenericInfoListView} view.

The following example shows how to access cell nodes and mesh information. It
iterates over all cells and calculates the barycenter for those that are in our
subdomain and are hexahedrons.

\snippet accelerator/SimpleHydroAcceleratorService.cc AcceleratorConnectivity

## Atomic Operations

The \arcaneacc{doAtomic} method allows performing atomic operations. The
supported operation types are defined by the \arcaneacc{eAtomicOperation}
enumeration. For example:

\snippet AtomicUnitTest.cc SampleAtomicAdd

## Advanced Algorithms: Reductions, Scan, Filtering, Partitioning, and Sorting

%Arcane offers several classes for performing more advanced algorithms. On the
accelerator, these algorithms generally use libraries provided by the
constructor ([CUB](https://nvidia.github.io/cccl/cub/index.html) for NVIDIA
and [rocprim](https://rocm.docs.amd.com/projects/rocPRIM/en/develop/reference/reference.html)
for AMD). The algorithms proposed by %Arcane therefore have the same limitations
as the underlying constructor implementation.

The available classes are:

- \arcaneacc{GenericFilterer} to filter elements of an array.
- \arcaneacc{GenericScanner} to perform inclusive or exclusive scan algorithms
  (see [Scan Algorithms](https://en.wikipedia.org/wiki/Prefix_sum) on Wikipedia)
- \arcaneacc{GenericSorter} to sort elements of a list
- \arcaneacc{GenericPartitioner} to partition elements of a list
- \arcaneacc{GenericReducer} to perform reductions. There are also other ways to
  perform reductions described on the page
  (\ref arcanedoc_acceleratorapi_reduction)

## Standalone Accelerator Mode {#arcanedoc_parallel_accelerator_standalone}

It is possible to use %Arcane's accelerator mode without support for high-level
objects such as meshes or subdomains.

In this mode, it is possible to use the %Arcane accelerator API directly from
the `main()` function, for example. To use this mode, simply use the class
method \arcane{ArcaneLauncher::createStandaloneAcceleratorMng()} after
initializing %Arcane:

```cpp
Arcane::ArcaneLauncher::init(Arcane::CommandLineArguments(&argc, &argv));
Arcane::StandaloneAcceleratorMng launcher(Arcane::ArcaneLauncher::createStandaloneAcceleratorMng());
```

The `launcher` instance must remain valid as long as you wish to use the
accelerator API. It is therefore preferable to define it in the code's `main()`.
The \arcane{StandaloneAcceleratorMng} class uses a reference semantics.
Therefore, it is possible to keep a reference to the instance anywhere in the
code if necessary.

The 'standalone_accelerator' example shows such usage. For example, the
following code allows offloading the sum of two arrays `a` and `b` into an array
`c` on the accelerator.

\snippet standalone_accelerator/main.cc StandaloneAcceleratorFull

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_acceleratorapi
</span>
<span class="next_section_button">
\ref arcanedoc_accelerator_materials
</span>
</div>
