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
    Name of the service used to manage parallelism. This variable
    must be set if you want a specific parallel execution model. Possible
    values are: `Mpi` or `Sequential`. The `Mpi` mode triggers the call
    to MPI_Init() and therefore the program must be launched by the
    platform's mpi launcher (e.g., `mpiexec`, `mpirun`, `prun`, ...).
    This variable should no longer be used because %Arcane automatically
    detects MPI launching. If you want to override the parallelism
    service, you must specify it in the command line arguments (see
    \ref arcanedoc_execution_launcher)
  </td>
</tr>
<tr>
  <td>
    ARCANE_CHECK
  </td>
  <td>
    If set, it adds verification tests for the validity of the
    operations performed. This is useful if the code crashes within an
    Arcane method. You can rerun the code with this variable in hopes of
    detecting the cause of the error. The 'check' version of %Arcane
    automatically includes these tests as well as other more costly
    tests like array overflows.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PAUSE_ON_ERROR
  </td>
  <td>
    If set, it pauses the code when an error is detected by
    %Arcane, such as a fatal exception or an array overflow.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PROFILING
  </td>
  <td>
    Name of the service to use for profiling information. Setting this
    option activates profiling. This allows you to see the time spent
    in each function at the end of execution. The two supported values
    are 'Papi' and 'Prof'. For 'Papi', the 'papi' library must be
    installed and the Linux kernel must be compatible. 'Prof' uses
    system signals.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PROFILING_PERIOD
  </td>
  <td>
    Number of CPU cycles between two profiling samples. This
    variable is only used if profiling is active with 'Papi'. A correct
    value is 500000, which on a 3GHz machine results in about 6000
    events per second.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PROFILING_ONLYFLOPS
  </td>
  <td>
    If true, it uses profiling only to determine the number of
    floating-point operations performed. It should be noted that this
    is the number of instructions, not the number of operations. On
    certain machines like Itaniums, one instruction can perform
    two operations (multiplication+addition). This option is only
    available if profiling is active with the 'Papi' option.
  </td>
</tr>
<tr>
  <td>
    ARCANE_MESSAGE_PASSING_PROFILING
  </td>
  <td>
    Allows activation of internal profiling for message passing
    operations. Possible values are:
    - **JSON**: uses the tracing service in the same format. This
      contains information about the time spent in each MPI function.
      The information is grouped by iteration and by time loop entry
      point.
    - **OTF2**, uses the tracing service in the same format. Once
      opened with an appropriate tool (e.g., Vampir), the details of
      MPI communications can be analyzed. The information allows
      identifying the MPI functions implemented at each time loop
      entry point, as well as those invoked by %Arcane variable
      synchronization operations.
  </td>
</tr>
<tr>
  <td>
    ARCANE_REDIRECT_SIGNALS
  </td>
  <td>
    Activates (TRUE) or deactivates (FALSE) signal redirection by
    Arcane. This redirection is active by default in Arcane and
    allows displaying the call stack, especially in case of a crash.
    However, this can interfere with other libraries, so it is
    possible to disable signal redirection.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PARALLEL_OUTPUT
  </td>
  <td>
    Activates (TRUE) or deactivates (FALSE) parallel listing outputs
    from all subdomains. Each subdomain will write the listing
    information to the file 'output#x', where #x is the subdomain
    number.
  </td>
</tr>
<tr>
  <td>
    ARCANE_TRACE_VARIABLE_CREATION
  </td>
  <td>
    If set, it retrieves the call stack for every variable reference
    creation. This information is then displayed at the end of
    execution to identify variables that were not deallocated. Note
    that this can significantly slow down execution.
  </td>
</tr>
<tr>
  <td>
    STDENV_VERIF, STDENV_VERIF_ENTRYPOINT, STDENV_VERIF_PATH, STDENV_VERIF_SKIP_GHOSTS
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
    See section \ref arcanedoc_debug_perf_check_memory
  </td>
</tr>
<tr>
  <td>
    ARCANE_DATA_INIT_POLICY
  </td>
  <td>
    Allows specifying the variable initialization policy. Using this
    variable is equivalent to calling the method
    Arcane::setGlobalDataInitialisationPolicy(). Possible values are:
    - **NONE**: no initialization
    - **DEFAULT**: default type value, which is 0 for integers and
      0.0 for reals
    - **NAN**: initializes reals with the NotANumber value, which
      allows stopping the code if you activate floating-point
      exceptions and use a variable not explicitly initialized.
    - **LEGACY**: mode prior to version 2.0 of %Arcane. It should
      no longer be used (see Arcane::DIP_Legacy).
  </td>
</tr>
<tr>
  <td>
    ARCANE_TEST_CLEANUP_AFTER_RUN
  </td>
  <td>
    Environment variable allowing the deletion of files generated by
    each test. Variable used by `arcane_test_driver`.
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
    Environment variable allowing multiple processes to write curves.
    With the historical API (Arcane::ITimeHistoryMng), all processes write
    the curves, and with the writers available in %Arcane, these processes
    write to the same file. A custom writer must therefore handle this.
    With the new APIs (Arcane::GlobalTimeHistoryAdder and Arcane::MeshTimeHistoryAdder),
    curve writing by multiple processes is handled correctly. This variable
    environment remains necessary to activate this feature (for now).
    See page \ref arcanedoc_io_timehistory_howto
  </td>
</tr>
<tr>
  <td>
    ARCANE_ENABLE_ALL_REPLICATS_WRITE_CURVES
  </td>
  <td>
    Environment variable useful when using replicas and allowing all
    processes to write curves.
  </td>
</tr>
<tr>
  <td>
    ARCANE_REPLACE_SYMBOLS_IN_DATASET
  </td>
  <td>
    Environment variable allowing symbol replacement in the dataset
    from command line arguments. See page \ref arcanedoc_execution_commandlineargs
  </td>
</tr>
<tr>
  <td>
    ARCANE_PARTICLE_NO_UNIQUE_ID_MAP
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>
<tr>
  <td>
    ARCANE_OLD_EXCHANGE
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>
<tr>
  <td>
    ARCANE_OLD_VARIABLE_SYNC
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>
<tr>
  <td>
    ARCANE_CHANGE_OWNER_ON_INIT
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>
<tr>
  <td>
    ARCANE_NB_EXCHANGE
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>
<tr>
  <td>
    ARCANE_CHECK_EXCHANGE
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>
<tr>
  <td>
    ARCANE_DEBUG_TIED_INTERFACE
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>
<tr>
  <td>
    ARCANE_THREAD_IMPLEMENTATION
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>
<tr>
  <td>
    ARCANE_LISTENER_TIMEOUT
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>
<tr>
  <td>
    ARCANE_VERIF_PARALLEL
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>
<tr>
  <td>
    ARCANE_FORCE_PREPARE_DUMP
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>
<tr>
  <td>
    ARCANE_SERIALIZE_USE_DERIVED_TYPE
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>
<tr>
  <td>
    ARCANE_TRACE_MPI
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>
<tr>
  <td>
    ARCANE_PARALLEL_CHECK_SYNC
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>
<tr>
  <td>
    ARCANE_TRACE_FUNCTION
  </td>
  <td>
    Internal use by %Arcane
  </td>
</tr>

</table>
