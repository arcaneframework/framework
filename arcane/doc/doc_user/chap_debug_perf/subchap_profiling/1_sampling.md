# Performance analysis by sampling {#arcanedoc_debug_perf_profiling_sampling}

[TOC]

\warning Currently, profiling only works on Linux platforms.

\warning Currently, profiling DOES NOT WORK when multi-threading (whether using
the task mechanism or message passing) is active.

Profiling in %Arcane works on a sampling principle: at regular intervals, the
code is interrupted and we check which method we are in. If the sampling
interval is small and the code is executed for a sufficient amount of time, we
obtain a good statistical representation of the proportion of time spent in the
most costly methods. Sampling methods do not require code instrumentation and
therefore slow down execution very little. However, they do not allow us to
know, for example, how many times a method is called, nor do they easily allow
us to know the call graph. For this, it is necessary to use mechanisms like
gprof with compiler support.


To activate profiling, you must set the ARCANE_PROFILING environment variable to
one of the following values:
- \a Prof . This allows for imprecise profiling but works on all Linux machines.
  The sampling frequency is around 20Hz, so you must run the code for at least 1
  minute to get significant results
- \a Papi. This allows for very precise profiling, using the processor's
  hardware counters. For this, the free PAPI library is used. This only works
  with recent Linux kernels (2.6.32 or later) or patched kernels, and %Arcane
  must be compiled with this support. In this mode, sampling is given in the
  number of processor clock cycles. The default value is 500,000 cycles. For a
  3GHz processor, this means 6000 samples per second. It is possible to change
  the number of cycles via the ARCANE_PROFILING_PERIOD environment variable. It
  is preferable not to go below the default value.

In order to maintain reasonable performance during execution, it is preferable
not to exceed 100,000 samples. Therefore, you must adjust the test execution
duration or the sampling frequency accordingly.

When active, profiling starts at the first iteration and stops at the last
iteration of the execution. In parallel, profiling is performed for each
processor, and %Arcane then displays the profiling information for each
processor in the following manner:

```log
*I-Internal    PROCESS_ID = 17977
*I-Internal    NB ADDRESS MAP = 737
*I-Internal    NB FUNC MAP = 53
*I-Internal    NB STACK MAP = 0
*I-Internal    TOTAL STACK = 0
*I-Internal    FUNC EVENT=2798
*I-Internal    FUNC FP=0
*I-Internal    TOTAL EVENT  = 2799
*I-Internal    TOTAL FP     = 1 (nb_giga_flip=0.0005)
*I-Internal    RATIO FP/CYC = 0.000357270453733476
*I-Internal    event     %   function
*I-Internal      1113   39.7      1113      0      0 0  SimpleHydro::ModuleSimpleHydro::computeCQs(Arcane::Real3*, Arcane::Real3*, Arcane::Cell const&)
*I-Internal       613   21.9       613      0      0 0  SimpleHydro::ModuleSimpleHydro::_computeGeometricValues(Arcane::ItemVectorViewT<Arcane::Cell>)
*I-Internal       396   14.1       396      0      0 0  SimpleHydro::ModuleSimpleHydro::_computePressureAndCellPseudoViscosityForces()
*I-Internal       217    7.7       217      0      0 0  SimpleHydro::ModuleSimpleHydro::_applyEquationOfState(Arcane::ItemVectorViewT<Arcane::Cell>)
*I-Internal        87    3.1        87      0      0 0  SimpleHydro::ModuleSimpleHydro::applyBoundaryCondition()
*I-Internal        85    3.0        85      0      0 0  SimpleHydro::ModuleSimpleHydro::_computeViscosityWork(Arcane::ItemVectorViewT<Arcane::Cell>)
*I-Internal        77    2.7        77      0      0 0  SimpleHydro::ModuleSimpleHydro::updateDensity()
*I-Internal        70    2.5        70      0      0 0  SimpleHydro::ModuleSimpleHydro::computeVelocity()
*I-Internal        50    1.7        50      0      0 0  SimpleHydro::ModuleSimpleHydro::computeDeltaT()
*I-Internal        28    1.0        28      0      0 0  SimpleHydro::ModuleSimpleHydro::moveNodes()
*I-Internal        13    0.4        13      0      0 0  Arcane::VariableArrayT<Arcane::Real3>::fill(Arcane::Real3 const&, Arcane::ItemGroup const&)
*I-Internal         4    0.1         4      0      0 0  _IO_vfprintf
*I-Internal         2    0.0         2      0      0 0  Arcane::Timer::Sentry::~Sentry()
*I-Internal         1    0.0         1      0      0 0  SimpleHydro::ModuleSimpleHydro::applyEquationOfState()
```

The first lines are internal information for %Arcane. Then, each method is
displayed in descending order of time spent. In the previous example, we see
that 39.7% of the time is spent in the ModuleSimpleHydro::computeCQs() method.

For the results to be relevant, the code must be compiled in optimized mode with
inlining enabled.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf_profiling
</span>
<span class="next_section_button">
\ref arcanedoc_debug_perf_profiling_mpi
</span>
</div>
