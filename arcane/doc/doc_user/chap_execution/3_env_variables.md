# Environment Variables {#arcanedoc_execution_env_variables}

The following environment variables allow you to modify the runtime behavior:

<table>
<tr>
  <th>Variable</th>
  <th>Description</th>
</tr>

<tr>
  <td>
    ARCANE_PARALLEL_SERVICE (obsolete)
  </td>
  <td>
    Name of the service used to manage parallelism. This variable must be set
  if you want a specific parallel execution model. Possible values are: `Mpi` or
  `Sequential`. The `Mpi` mode triggers the call to MPI_Init() and therefore the
  program must be launched by the platform's mpi launcher (e.g., `mpiexec`,
  `mpirun`, `prun`, ...). This variable should no longer be used because %Arcane
  automatically detects MPI launching. If you want to override the parallelism
  service, you must specify it in the command line arguments (see
  \ref arcanedoc_execution_launcher)
  </td>
</tr>
<tr>
  <td>
    ARCANE_CHECK
  </td>
  <td>
    If defined, it adds verification tests for the validity of the operations
    performed. This is useful if the code crashes within an Arcane method. It is
    possible to rerun the code with this variable in hopes of detecting the
    cause of the error. The 'check' version of %Arcane automatically includes
    these tests as well as other more costly tests like array overflows.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PAUSE_ON_ERROR
  </td>
  <td>
    If defined, it pauses the code when an error is detected by %Arcane, such as
    a fatal exception or an array overflow.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PROFILING
  </td>
  <td>
    Name of the service to use for profiling information. Setting this option
    activates profiling. This allows you to see the time spent in each function
    at the end of execution. The two supported values are 'Papi' and 'Prof'. For
    'Papi', the 'papi' library must be installed and the Linux kernel must be
    compatible. 'Prof' uses system signals.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PROFILING_PERIOD
  </td>
  <td>
    Number of CPU cycles between two profiling samples. This variable is only
    used if profiling is active with 'Papi'. A correct value is 500000, which
    on a 3Ghz machine results in about 6000 events per second.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PROFILING_ONLYFLOPS
  </td>
  <td>
    If true, it uses profiling solely to determine the number of floating-point
    operations performed. It should be noted that this is the number of
    instructions, not the number of operations. On certain machines like
    Itaniums, one instruction can perform two operations (multiplication +
    addition). This option is only available if profiling is active with the
    'Papi' option.
  </td>
</tr>
<tr>
  <td>
    ARCANE_LOOP_PROFILING_LEVEL
  </td>
  <td>
    (version 3.8+). Integer indicating the loop profiling level. Setting it to
    a value greater than or equal to 1 activates loop traces based on macros
    provided by %Arcane such as ENUMERATE_() (ENUMERATE_CELL(),
    ENUMERATE_NODE(), ...), and RUNCOMMAND (RUNCOMMAND_LOOP(),
    RUNCOMMAND_ENUMERATE()). At the end of the calculation, the time spent in
    each loop will be written in the listing and in files.
  </td>
</tr>

<tr>
  <td>
    ARCANE_MESSAGE_PASSING_PROFILING
  </td>
  <td>
    Allows activation of internal profiling for message passing operations.
    Possible values are:
    - **JSON**: uses the tracing service in the same format. This contains
      information on the time spent in each MPI function. The information is
      grouped by iteration and by time loop entry point.
    - **OTF2**, uses the tracing service in the same format. Once opened with
      a suitable tool (e.g., Vampir), the details of MPI communications can be
      analyzed. The information notably allows identifying the MPI functions
      implemented at each time loop entry point as well as those invoked by
      %Arcane variable synchronization operations.
  </td>
</tr>
<tr>
  <td>
    ARCANE_REDIRECT_SIGNALS
  </td>
  <td>
    Activates (TRUE) or deactivates (FALSE) signal redirection by Arcane. This
    redirection is active by default in Arcane and allows displaying the call
    stack, especially in case of a crash. Nevertheless, this can interfere with
    other libraries, so it is possible to disable signal redirection.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PARALLEL_OUTPUT
  </td>
  <td>
    Activates (TRUE) or deactivates (FALSE) the listing outputs of all
    subdomains in parallel. Each subdomain will write the listing information to
    the file 'output#x' where #x is the subdomain number.
  </td>
</tr>
<tr>
  <td>
    ARCANE_TRACE_VARIABLE_CREATION
  </td>
  <td>
    If defined, it retrieves the call stack for every variable reference
    creation. This information is then displayed at the end of execution to
    identify variables that have not been deallocated. Note that this can
    significantly slow down execution.
  </td>
</tr>
<tr>
  <td>
    STDENV_VERIF, STDENV_VERIF_ENTRYPOINT, STDENV_VERIF_PATH,
    STDENV_VERIF_SKIP_GHOSTS
  </td>
  <td>
    See section \ref arcanedoc_debug_perf_compare_bittobit
  </td>
</tr>
<tr>
  <td>
    ARCANE_CHECK_MEMORY, ARCANE_CHECK_MEMORY_BLOCK_SIZE
  </td>
  <td>
    See section \ref arcanedoc_debug_perf_check_memory. This only works with
    certain versions of Linux and glibc. This mechanism uses obsolete glibc
    features and is therefore not available with recent versions (>2021) of
    glibc. Notably, ubuntu 22.04 does not support this mechanism.
  </td>
</tr>
<tr>
  <td>
    ARCANE_DATA_INIT_POLICY
  </td>
  <td>
    Allows specifying the variable initialization policy. Using this variable
    is equivalent to calling the method
    Arcane::setGlobalDataInitialisationPolicy(). Possible values are:
    - **NONE**: no initialization
    - **DEFAULT**: default type value, i.e., 0 for integers and 0.0 for reals
    - **NAN**: initializes reals with the NotANumber value, which allows, if
      floating-point exceptions are enabled, stopping the code if an explicitly
      uninitialized variable is used.
    - **LEGACY**: mode prior to version 2.0 of %Arcane. It should no longer be
      used (see Arcane::DIP_Legacy).
  </td>
</tr>
<tr>
  <td>
    ARCANE_LISTENER_TIMEOUT
  </td>
  <td>
    todo
  </td>
</tr>
<tr>
  <td>
    ARCANE_GDB_STACK
  </td>
  <td>
    todo
  </td>
</tr>
<tr>
  <td>
    ARCANE_ENABLE_NON_IO_MASTER_CURVES
  </td>
  <td>
    Environment variable allowing multiple processes to write curves. With the
    historical API (Arcane::ITimeHistoryMng), all processes write the curves,
    and with the writers available in %Arcane, these processes write to the same
    file. A custom writer must therefore handle this. With the new APIs
    (Arcane::GlobalTimeHistoryAdder and Arcane::MeshTimeHistoryAdder), writing
    curves by multiple processes is handled correctly. This environment variable
    remains necessary to activate this feature (for now). See page
    \ref arcanedoc_io_timehistory_howto
  </td>
</tr>
<tr>
  <td>
    ARCANE_ENABLE_ALL_REPLICATS_WRITE_CURVES
  </td>
  <td>
    Environment variable useful when using replicas and allowing all processes
    to write curves.
  </td>
</tr>
<tr>
  <td>
    ARCANE_REPLACE_SYMBOLS_IN_DATASET
  </td>
  <td>
    Environment variable allowing the replacement of symbols in the dataset from
    command line arguments. See page \ref arcanedoc_execution_commandlineargs
  </td>
</tr>
<tr>
  <td>
    ARCANE_USE_BACKWARDCPP
  </td>
  <td>
    Environment variable allowing the display of the call stack via
    BackwardCpp.<br>
    BackwardCpp requires Arcane to be compiled with at least the DW library.<br>
    Default value: **0**<br>
    Possible values: **0** or **1**
  </td>
</tr>
<tr>
  <td>
    ARCANE_CALLSTACK_VERBOSE
  </td>
  <td>
    Environment variable allowing more or less information to be displayed when
    showing the call stack.<br>
    Requires the ARCANE_USE_BACKWARDCPP environment variable.<br>
    Default value: **2**<br>
    Possible values:<br>
    - **0**: Classic CallStack (function name only)
    - **1**: Classic CallStack with line number and file for classes/functions outside the Arcane namespace
    - **2**: Classic CallStack with line number and file for all classes/functions
    - **3**: Classic CallStack with line number, file for all classes/functions and snippet for classes/functions outside the Arcane namespace
    - **4**: Classic CallStack with line number, file and snippet for all classes/functions
  </td>
</tr>
<tr>
  <td>
    ARCANE_CALLSTACK_HUMAN_READABLE
  </td>
  <td>
    Environment variable allowing the call stack to be displayed with spaces
    between calls and the line number before the source file path, to improve
    readability.<br>
    If the value is set to zero, the file path and line number are displayed in
    a format recognizable by a debugger or an IDE (path:line).<br>
    Requires the ARCANE_USE_BACKWARDCPP environment variable.<br>
    Default value: **1**<br>
    Possible values: **0** or **1**
  </td>
</tr>


</table>


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_execution_direct_execution
</span>
<span class="next_section_button">
\ref arcanedoc_execution_traces
</span>
</div>
