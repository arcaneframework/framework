# Configuration File {#arcanedoc_examples_concret_example_config}

[TOC]

To start, here is the configuration file:

```xml
<?xml version="1.0" ?>
<arcane-config code-name="Quicksilver">
  <time-loops>
    <time-loop name="QAMALoop">
      <title>QS</title>
      <description>Default timeloop for code Quicksilver Arcane MiniApp
      </description>

      <singleton-services>
        <service name="SimpleCsvOutput" need="required"/>
        <service name="SimpleCsvComparator" need="required"/>
        <service name="RNG" need="required"/>
      </singleton-services>

      <modules>
        <module name="QS" need="required"/>
        <module name="SamplingMC" need="required"/>
        <module name="TrackingMC" need="required"/>
      </modules>

      <entry-points where="init">
        <entry-point name="QS.InitModule"/>
        <entry-point name="SamplingMC.InitModule"/>
        <entry-point name="TrackingMC.InitModule"/>
        <entry-point name="QS.StartLoadBalancing"/>
      </entry-points>

      <entry-points where="compute-loop">
        <entry-point name="SamplingMC.CycleSampling"/>
        <entry-point name="TrackingMC.CycleTracking"/>
        <entry-point name="QS.CycleFinalize"/>
        <entry-point name="SamplingMC.CycleFinalize"/>
        <entry-point name="TrackingMC.CycleFinalize"/>
        <entry-point name="QS.LoopLoadBalancing"/>
      </entry-points>

      <entry-points where="on-mesh-changed">
        <entry-point name="QS.AfterLoadBalancing"/>
      </entry-points>

      <entry-points where="exit">
        <entry-point name="SamplingMC.EndModule"/>
        <entry-point name="TrackingMC.EndModule"/>
        <entry-point name="QS.CompareWithReference"/>
        <entry-point name="QS.EndModule"/>
      </entry-points>

    </time-loop>
  </time-loops>
</arcane-config>
```
With this file, we can already see what `QAMA` looks like.
We find our three modules `QS`, `SamplingMC`, and `TrackingMC` in the`<modules>`
section.
We also find the three types of entry points that we saw in the `HelloWorld`
example: `init`, `compute-loop`, and `exit` (here:
\ref arcanedoc_examples_simple_example_module_sayhelloaxl).

\note
In `HelloWorld`, there was only one entry point per entry point type, so there
was no need to worry about the order. Here, we have several. It is therefore
important to note that the order of entry points is important and is taken into
account. Conversely, the order of entry point types is not important.
```xml
<!-- 1) -->
<entry-points where="compute-loop">
  <entry-point name="SamplingMC.CycleSampling" />
  <entry-point name="TrackingMC.CycleTracking" />
</entry-points>

<entry-points where="init">
  <entry-point name="QS.InitModule" />
  <entry-point name="SamplingMC.InitModule" />
</entry-points>
```
gives the same result as:
```xml
<!-- 2) -->
<entry-points where="init">
  <entry-point name="QS.InitModule" />
  <entry-point name="SamplingMC.InitModule" />
</entry-points>

<entry-points where="compute-loop">
  <entry-point name="SamplingMC.CycleSampling" />
  <entry-point name="TrackingMC.CycleTracking" />
</entry-points>
```
but is different from:
```xml
<!-- 3) -->
<entry-points where="init">
  <entry-point name="SamplingMC.InitModule"/> <!-- here -->
  <entry-point name="QS.InitModule"/>         <!-- here -->
</entry-points>

<entry-points where="compute-loop">
  <entry-point name="SamplingMC.CycleSampling" />
  <entry-point name="TrackingMC.CycleTracking" />
</entry-points>
```


____

In the new features, we first have the `<singleton-services>` section.
We find the service seen in the previous section: `RNG`. We also have
two other services: `Arcane::SimpleCsvOutput` and `Arcane::SimpleCsvComparator`.
These services are services included in the %Arcane framework and can
therefore be used by any application.

Their interfaces are available in this documentation:
`Arcane::ISimpleTableOutput`, `Arcane::ISimpleTableComparator`, and
`Arcane::IRandomNumberGenerator`.

Like the `RNG` service, it is possible to create an implementation
specific to our application using the interface of these services.

____

There are two ways to use a service:
- as a normal service, which must be declared in the `.axl` of a module. In
  this case, there will be one object per module that declares it, and it will
  not be shared.
- as a singleton, which must be declared in the project's `.config`.
  In this case, there will only be one object per project. Modules can
  retrieve a pointer to this unique object.

In QAMA, I chose the singleton method. For the `SimpleCsvOutput` service,
it is necessary to share the object to generate a single CSV table.
For the `RNG` service, it doesn't matter.

\warning In the case of a singleton, it is impossible to retrieve data
from the dataset autonomously. However, it is possible to pass the data through
one of the present modules. In the case of Quicksilver,
it is the QSModule that is responsible for doing this.
It is possible to determine, for a service, whether it is considered a singleton
with this line:
```cpp
option() == null;
```
If there are options, then we are in a classic service; otherwise, we are in a
singleton.

____

We also have a new type of entry point: `on-mesh-changed`.
This type of entry point is triggered when the mesh changes,
for example, during a redistribution.
This entry point executes after `compute-loop` type entry points.
\warning
In practice, if you retrieve `m_global_iteration()` in a `compute-loop` type
entry point, you will get iteration **i** (example: `QS.LoopLoadBalancing`),
then in the next `on-mesh-changed` type entry point, you will get iteration
**i+1** (example: `QS.AfterLoadBalancing`).

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_concret_example_struct
</span>
<span class="next_section_button">
\ref arcanedoc_examples_concret_example_rng
</span>
</div>
