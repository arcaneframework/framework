# Launching a Calculation {#arcanedoc_execution_launcher}

<!-- [TOC] -->

There are two mechanisms for executing code with %Arcane:

1. the time loop mechanism, which is the classic mechanism available since the
   first versions of %Arcane.
2. direct execution

For both mechanisms, you must use the class Arcane::ArcaneLauncher. This class
allows you to specify the execution parameters. All methods of this class are
static.

The first thing to do is call the method Arcane::ArcaneLauncher::init() to
specify the execution parameters to %Arcane. This allows certain command-line
values (such as the verbosity level, the output directory name, ...) to be
automatically analyzed.

```cpp
#include <arcane/launcher/ArcaneLauncher.h>

using namespace Arcane;

int
main(int argc,char* argv[])
{
  ArcaneLauncher::init(Arcane::CommandLineArguments(&argc,&argv));
  ...
}
```

There are two classes for specifying execution parameters:
Arcane::ApplicationInfo and Arcane::ApplicationBuildInfo.

\note These two classes exist for compatibility reasons with existing code.
Eventually, only the Arcane::ApplicationBuildInfo class will remain, so this is
the one that must be used.

The static instances of these two classes can be retrieved via the methods
Arcane::ArcaneLauncher::applicationInfo() and
Arcane::ArcaneLauncher::applicationBuildInfo(). The following example shows how
to change the code name and version and the default directory for outputs:

```cpp
#include <arcane/launcher/ArcaneLauncher.h>

using namespace Arcane;

int
main(int argc,char* argv[])
{
  ArcaneLauncher::init(Arcane::CommandLineArguments(&argc,&argv));
  Arcane::ApplicationBuildInfo& app_build_info = ArcaneLauncher::applicationBuildInfo();
  app_build_info.setCodeName("ArcaneTest");
  app_build_info.setCodeVersion(VersionInfo(1,0,0));
  app_build_info.setOutputDirectory("test_output");
  ...
}
```

Once initialization is complete, it is possible to launch the code execution via
the call to Arcane::ArcaneLauncher::run():

```cpp
#include <arcane/launcher/ArcaneLauncher.h>

using namespace Arcane;

int
main(int argc,char* argv[])
{
  ArcaneLauncher::init(Arcane::CommandLineArguments(&argc,&argv));
  Arcane::ApplicationBuildInfo& app_build_info = ArcaneLauncher::applicationBuildInfo();
  app_build_info.setCodeName("ArcaneTest");
  app_build_info.setCodeVersion(VersionInfo(1,0,0));
  app_build_info.setOutputDirectory("test_output");
  return ArcaneLauncher::run();
}
```

## MPI Initialization {#arcanedoc_execution_launcher_mpi}

Using MPI requires calling the `MPI_Init_thread()` method from the MPI library.
If %Arcane is compiled with MPI support, MPI detection and the call to
`MPI_Init_thread()` are done automatically by %Arcane. The thread support level
used for the call to `MPI_Init_thread()` depends on options such as the number
of tasks or local subdomains in hybrid mode that you wish to use.

However, it is possible for the code to initialize MPI itself if it wishes. To
do this, it must call the method `MPI_Init_thread()` before calling
Arcane::ArcaneLauncher::run().

\note Even if the executable is used sequentially (i.e., without going through a
command such as `mpiexec ...`), %Arcane attempts to initialize MPI. This is
necessary because certain libraries (for example, linear solvers) require MPI to
be initialized in all cases. It is possible to modify this behavior by
explicitly specifying the desired parallelism service (TODO document).

\warning You must be careful to use the `mpiexec` executable that corresponds to
the version of MPI with which %Arcane was compiled, otherwise you will run the
sequential execution *N* times.

## Code Execution {#arcanedoc_execution_launcher_exec}

The method Arcane::ArcaneLauncher::run() allows you to launch the code
execution. This method has three overloads:

1. The call without arguments (Arcane::ArcaneLauncher::run()) to launch the
   classic execution using a time loop (see
   \ref arcanedoc_core_types_codeconfig). This mechanism should be preferred
   because it allows you to use all of %Arcane's features. The page
   \ref arcanedoc_execution_launcher shows a minimal example of this type of
   usage.
2. Arcane::ArcaneLauncher::run(std::function<int(DirectSubDomainExecutionContext&)> func)
   to execute the code specified by \a func after initialization and subdomain
   creation. The page \ref arcanedoc_general_direct_execution shows an example
   of direct execution.
3. Arcane::ArcaneLauncher::run(std::function<int(DirectExecutionContext&)> func)
   to execute the code specified by \a func **sequentially** only. This
   mechanism should be used if, for example, you want to perform simple unit
   tests without having a subdomain (Arcane::ISubDomain*). application without
   dataset or time loop.

## Command Line Options {#arcanedoc_execution_launcher_options}

%Arcane interprets command-line options that start with `-A`. For example, to
change the verbosity level, simply specify the option `-A,VerbosityLevel=3` in
the command line.

The options are interpreted when calling Arcane::ArcaneLauncher::init(), and the
values of ArcaneLauncher::applicationBuildInfo() are automatically filled with
these options. However, it is possible to override them if necessary.

\remark It is also possible to modify the dataset using command-line options.
This possibility is discussed on the page
\ref arcanedoc_execution_commandlineargs.

The available options are:

<table>
<tr>
<th>Option</th>
<th>Environment Variable</th>
<th>Type</th>
<th>Default</th>
<th>Description</th>
</tr>

<tr>
<td>T</td>
<td>ARCANE_NB_TASK</td>
<td>Int32</td>
<td>1</td>
<td>Number of concurrent tasks to execute</td>
</tr>

<tr>
<td>S</td>
<td>ARCANE_NB_THREAD (This environment variable is obsolete)</td>
<td>Int32</td>
<td></td>
<td>Number of subdomains in shared memory</td>
</tr>

<tr>
<td>R</td>
<td>ARCANE_NB_REPLICATION (This environment variable is obsolete)</td>
<td>Int32</td>
<td>1</td>
<td>Number of replicated subdomains</td>
</tr>

<tr>
<td>P</td>
<td>ARCANE_NB_SUB_DOMAIN (This environment variable is obsolete)</td>
<td>Int32</td>
<td></td>
<td>Number of processes to use for subdomains. This value is normally calculated
automatically based on MPI parameters. It is only useful if you wish to use
fewer processes for domain partitioning than those allocated for the
calculation.
</td>
</tr>

<tr>
<td>AcceleratorRuntime</td>
<td></td>
<td>string</td>
<td></td>
<td>Accelerator runtime to use. The two possible values are `cuda` or `hip`. You
must have compiled %Arcane with accelerator support for this option to be
accessible.
</td>
</tr>

<tr>
<td>MaxIteration</td>
<td></td>
<td>Int32</td>
<td></td>
<td>Maximum number of iterations to perform for the execution. If the number of
iterations specified by this variable is reached, the calculation stops.
</td>
</tr>

<tr>
<td>OutputLevel</td>
<td>ARCANE_OUTPUT_LEVEL</td>
<td>Int32</td>
<td>3</td>
<td>Verbosity level of messages on standard output.
</td>
</tr>

<tr>
<td>VerbosityLevel</td>
<td>ARCANE_VERBOSITY_LEVEL</td>
<td>Int32</td>
<td>3</td>
<td>Verbosity level of messages for listing file outputs. If the `OutputLevel`
option is not specified, this option is also used for standard outputs.
</td>
</tr>

<tr>
<td>MinimalVerbosityLevel</td>
<td></td>
<td>Int32</td>
<td></td>
<td>Minimum verbosity level. If specified, explicit calls in the code to change
verbosity (via Arccore::ITraceMng::setVerbosityLevel()) cannot go below this
minimum verbosity level. This mechanism is mainly used for debugging to ensure
message display.
</td>
</tr>

<tr>
<td>MasterHasOutputFile</td>
<td>ARCANE_MASTER_HAS_OUTPUT_FILE</td>
<td>Bool</td>
<td>False</td>
<td>Indicates whether the master process (generally process 0) writes the
listing to a file in addition to standard output</td>
</tr>

<tr>
<td>OutputDirectory</td>
<td>ARCANE_OUTPUT_DIRECTORY</td>
<td>String</td>
<td>.</td>
<td>Base directory for generated files (listings, logs, curves, output, ...).
This value is the one returned by Arcane::ISubDomain::exportDirectory().
</td>
</tr>

<tr>
<td>CaseDatasetFileName</td>
<td></td>
<td>String</td>
<td></td>
<td>Dataset file name. If not specified and required, the last argument of the
command line is considered the dataset file name.
</td>
</tr>

<tr>
<td>ThreadBindingStrategy</td>
<td>ARCANE_THREAD_BINDING_STRATEGY</td>
<td>String</td>
<td></td>
<td>Thread binding strategy. This only works if %Arcane is compiled with the
'hwloc' library. By default, no binding is performed. The only available mode is
'Simple', which allocates threads according to a round-robin mechanism.

NOTE: this binding mechanism is under development and may not function optimally
in all cases
</td>
</tr>

<tr>
<td>ParallelLoopGrainSize</td>
<td></td>
<td>Int32</td>
<td></td>
<td>Grain size for multi-threaded parallel loops. If set, it indicates the
number of elements in each block that decomposes a multi-threaded loop (from
version 3.8).
</td>
</tr>

<tr>
<td>ParallelLoopPartitioner</td>
<td></td>
<td>String</td>
<td></td>
<td>Choice of partitioner for multi-threaded parallel loops. Possible values are
`auto`, `static`, or `deterministic` (from version 3.8).
</td>
</tr>

</table>

## Choosing the Message Exchange Manager {#arcanedoc_execution_launcher_exchange}

The message exchange manager (Arcane::IParallelSuperMng) is chosen when
launching the calculation. %Arcane provides the following managers:

- MpiParallelSuperMng
- SequentialParallelSuperMng
- MpiSequentialParallelSuperMng
- SharedMemoryParallelSuperMng
- HybridParallelSuperMng

Generally, %Arcane automatically chooses the manager based on the parameters
used to launch the calculation, but it is possible to explicitly specify the
manager to use by setting the environment variable (obsolete)
`ARCANE_PARALLEL_SERVICE` or by specifying the `MessagePassingService` option in
the command line with one of the values above (without the `ParallelSuperMng`
suffix, so for example `Mpi`, `Sequential`, `MpiSequential`, ...).

The automatic choice of the manager is made as follows:

<table>
<tr>
<th>Command Line</th>
<th>Manager Used</th>
<th>Description</th>
</tr>
<tr>
<td>`./a.out ...`</td>
<td>`MpiSequentialParallelSuperMng` or `SequentialParallelSuperMng`</td>
<td>`MpiSequentialParallelSuperMng` if %Arcane was compiled with MPI,
`SequentialParallelSuperMng` otherwise. The difference between the two is that
the former initializes MPI so that communicators such as `MPI_COMM_WORLD` can
be used
</td>
</tr>

<tr>
<td>`mpiexec -n $N ./a.out ...`</td>
<td>`MpiParallelSuperMng`</td>
<td>$N processes, 1 subdomain per process</td>
</tr>

<tr>
<td>`./a.out -A,S=$S ...`</td>
<td>`SharedMemoryParallelSuperMng`</td>
<td>1 process, $S subdomains per process. Communication between subdomains is
done via message exchange in shared memory.
</td>
</tr>

<tr>
<td>`mpiexec -n $N ./a.out -A,S=$S ...`</td>
<td>`HybridParallelSuperMng`</td>
<td>$N processes, $S subdomains per process, resulting in $N * $S subdomains in
total.
</td>
</tr>

</table>

Here are some launch examples:

```sh
# sequential launch of the dataset 'Test.arc'
a.out Test.arc

# launch with 4 MPI subdomains
mpiexec -n 4 a.out Test.arc

# launch with 4 subdomains in shared memory mode
a.out -A,S=4 Test.arc

# launch with 12 subdomains and 4 processes (4 subdomains in
# shared memory per process)
mpiexec -n 3 -c 4 a.out -A,S=4 Test.arc

# launch with the CUDA accelerator runtime.
a.out -A,AcceleratorRuntime=cuda Test.arc
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_execution
</span>
<span class="next_section_button">
\ref arcanedoc_execution_direct_execution
</span>
</div>
