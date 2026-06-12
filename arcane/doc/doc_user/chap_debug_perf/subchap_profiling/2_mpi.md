# Performance analysis via internal instrumentation {#arcanedoc_debug_perf_profiling_mpi}


\warning Currently, *profiling* only works on **Linux** platforms.


\warning Currently, *profiling* only works for the MPI implementation of
*message passing*.


\warning The *message passing* *profiling* features described below are still
under development.


%Arcane has an internal tracing mechanism for post-mortem performance analysis
of so-called *message passing* operations.
This mechanism is transparent to the user, who can activate this feature *via*
the environment variable **ARCANE_MESSAGE_PASSING_PROFILING**.
Depending on the value passed to it, one of the services will be executed:
- \ref arcanedoc_debug_perf_profiling_mpi_json
- \ref arcanedoc_debug_perf_profiling_mpi_otf2

Each of these services will provide traces in the associated format.


## JSON {#arcanedoc_debug_perf_profiling_mpi_json}


When the environment variable **ARCANE_MESSAGE_PASSING_PROFILING=JSON** is set,
the internal *message passing* profiling service in JSON format is activated.
It will monitor MPI functions by iteration and by entry point, indicating the
number of function calls, the size of the exchanged messages (in bytes), and the
time spent in them (in seconds).
This information is available in the case output listing subdirectory.
The files are named **message_passing_logs.i.json** where **i** corresponds to
the number of the monitored subdomain.


JSON trace example:

```json
{
  "1": {
    "ArcaneTimeLoopBegin": {
      "MPI_Allgather": {
        "Count": 2,
        "MessageSize": 4,
        "TotalTime": 0.0000171661376953125
      }
    },
  },
  "2": {
    "TP_testLoop": {
      "MPI_Allreduce": {
        "Count": 3,
        "MessageSize": 2014,
        "TotalTime": 0.034449100494384769
      },
      "MPI_Recv": {
        "Count": 22,
        "MessageSize": 307625744,
        "TotalTime": 0.00004458427429199219
      },
      "Synchronize": {
        "Count": 82,
        "MessageSize": 82,
        "TotalTime": 0.0026214122772216799
      }
    }
  }
}

```


\note A browser plugin allowing reading and manipulating traces provided in this
format is under study.

## OTF2 {#arcanedoc_debug_perf_profiling_mpi_otf2}


When the environment variable **ARCANE_MESSAGE_PASSING_PROFILING=OTF2** is set,
the internal *message passing* profiling service in OTF2 format is activated.
The OTF2 format is an open-source binary format (Open Trace Format 2) intended
to be read by profiling tools such as Vampir, Scalasca, Tau, Score-P, etc.

Each call to MPI functions is instrumented and allows detailed information to be
obtained about communications between subdomains, notably enabling the detection
of load imbalance issues or the identification of poor exchange patterns.

This information is available in the case output listing subdirectory.
They are stored in 2 files and one directory. The directory takes the name of
the case, and the two files do as well, modulo the *.otf2* and *.def*
extensions.
\note The files and the directory are overwritten with every new simulation
launch.


OTF2 trace example visualized by Vampir:

\image html ex_otf2_vampir.png


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf_profiling_sampling
</span>
<span class="next_section_button">
\ref arcanedoc_debug_perf_profiling_loop
</span>
</div>
