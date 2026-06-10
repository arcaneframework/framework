# Loop Performance Analysis {#arcanedoc_debug_perf_profiling_loop}

[TOC]

It is possible to profile loops managed by %Arcane. This applies to all loops
such as ENUMERATE_(), ENUMERATE_CELL(), RUNCOMMAND_ENUMERATE(),
RUNCOMMAND_LOOP(). Profiling works for sequential, multi-threaded, or
accelerator code. In this mode, the time taken to execute each loop is measured,
and the accumulated information about these loops is displayed at the end of
execution.

\note For performance reasons, profiling of loops such as ENUMERATE_() is not
active by default. To activate it, you must compile the code using these loops
with the macro `ARCANE_TRACE_ENUMERATOR`.

To get the profiling information, simply set the environment variable
`ARCANE_LOOP_PROFILING_LEVEL`. The two possible values are:

- `1` to activate basic profiling.
- `2` to activate profiling like mode `1`. The difference with this mode is that
  events are used to calculate the time spent in accelerator cores. This mode is
  more precise than mode `1` but may cause a slight overhead on calculation
  time (on the order of a percent).

It is also possible to set profiling programmatically by calling the method
\arcane{ProfilingRegistry::setProfilingLevel()}. It is possible to activate and
deactivate profiling at any time outside of loops.

When profiling is active, information is displayed at the end of the
calculation.

Here is an example of the result on an accelerator:

```
*I-Internal   LoopStatistics:
LoopStat: global_time (ms) = 42.7763
LoopStat: global_nb_loop   =        752 time=56883.4
LoopStat: global_nb_chunk  =          0 time=0
ProfilingStat
     Ncall    Nchunk     T (ms)  Tck (ns)     %  name
        51         0     21.805         0  50.9  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeGeometricValues()
       100         0      8.551         0  19.9  void SimpleHydro::SimpleHydroAcceleratorService::_computePressureAndCellPseudoViscosityForces()
       300         0      4.185         0   9.7  virtual void SimpleHydro::SimpleHydroAcceleratorService::applyBoundaryCondition()
        50         0      2.242         0   5.2  virtual void SimpleHydro::SimpleHydroAcceleratorService::applyEquationOfState()
        50         0      2.026         0   4.7  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeViscosityWork()
        50         0      1.522         0   3.5  virtual void SimpleHydro::SimpleHydroAcceleratorService::updateDensity()
        50         0      0.866         0   2.0  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeDeltaT()
        50         0      0.679         0   1.5  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeVelocity()
        50         0      0.557         0   1.3  virtual void SimpleHydro::SimpleHydroAcceleratorService::moveNodes()
         1         0      0.337         0   0.7  virtual void SimpleHydro::SimpleHydroAcceleratorService::hydroStartInit()
```

and an example in multi-thread:

```
*I-Internal   LoopStatistics:
LoopStat: global_time (ms) = 141.137
LoopStat: global_nb_loop   =       1504 time=93841.2
LoopStat: global_nb_chunk  =      35028 time=4029.27
ProfilingStat
     Ncall    Nchunk     T (ms)  Tck (ns)     %  name
       102      3264     80.491     24660  57.0  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeGeometricValues()
       200      8000     34.268      4283  24.2  void SimpleHydro::SimpleHydroAcceleratorService::_computePressureAndCellPseudoViscosityForces()
       100      3200      7.032      2197   4.9  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeViscosityWork()
       100      3200      6.807      2127   4.8  virtual void SimpleHydro::SimpleHydroAcceleratorService::applyEquationOfState()
       100      4800      2.818       587   1.9  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeVelocity()
       100      4800      2.733       569   1.9  virtual void SimpleHydro::SimpleHydroAcceleratorService::moveNodes()
       600      1300      2.502      1925   1.7  virtual void SimpleHydro::SimpleHydroAcceleratorService::applyBoundaryCondition()
       100      3200      2.410       753   1.7  virtual void SimpleHydro::SimpleHydroAcceleratorService::updateDensity()
       100      3200      1.875       586   1.3  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeDeltaT()
         2        64      0.196      3064   0.1  virtual void SimpleHydro::SimpleHydroAcceleratorService::hydroStartInit()
```


Here is the meaning of the fields:

- `Ncall` : number of times the loop is executed
- `Nchunk` : number of loop partitions (chunks) in multi-thread mode.
- `T` : total time (in milliseconds) spent executing the loop. In multi-thread,
  this is the total time accumulated across all threads.
- `Tck` : time per chunk. This value is only valid for multi-thread executions.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf_profiling_mpi
</span>
</div>
