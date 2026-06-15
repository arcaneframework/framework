# Entrypoint {#arcanedoc_core_types_axl_entrypoint}

[TOC]

An entrypoint is a module method that is referenced by %Arcane and serves as an
interface for the module with the time loop. An entrypoint is described by the
interface class \c IEntryPoint. An entrypoint is a method with the following
signature, where <b>T</b> is the type of the module class:

```cpp
void T::func();
```

An entrypoint is characterized by:
- a name
- a method of the associated class.
- the location where it can be called (initialization, computation loop, ...).
  By default, an entrypoint is called in the computation loop.

Entrypoints are declared in the module descriptor. For example, for the \c Test
module, 2 entrypoints are declared which can then be called in the time loop by
the names <b>DumpConnection</b> or <b>TestPressureSync</b>:

```xml
<module name="Test" version="1.0">
  <name lang="fr">Test</name>

  <description>Descripteur du module Test</description>

  <variables>
    <!-- .... cf chapitre sur les variables .... -->
  </variables>

  <entry-points>
    <entry-point method-name="testPressureSync" name="TestPressureSync"
                 where="compute-loop" property="none"/>
    <entry-point method-name="dumpConnection" name="DumpConnection"
                 where="compute-loop" property="none"/>
  </entry-points>

  <options>
  </options>
</module>
```

The meaning of the attributes of the **entry-point** element is as follows:
- **method-name** defines the name of the C++ method corresponding to the
  entrypoint,
- **name** is the registration name of the entrypoint in %Arcane,
- **property** gives the type of entrypoint:
  - **none**: "traditional" entrypoint
  - **auto-load-begin**: means that the module for this entrypoint will be
    automatically loaded and the entrypoint will be called at the beginning of
    the time loop,
  - **auto-load-end**: means that the module for this entrypoint will be
    automatically loaded and the entrypoint will be called at the end of the
    time loop
- **where** is the location where the entrypoint will be called. Possible values
  are:

<table>
<tr>
<td> **compute-loop**</td>
<td> entrypoint called during loop iteration</td>
</tr>
<tr>
<td> **init** </td>
<td> used to initialize the module's data structures that are not preserved
during a checkpoint. At this stage of initialization, the dataset and the mesh
have already been read. Initialization is also used to check certain values,
calculate initial values...</td>
</tr>
<tr>
<td> **start-init** </td>
<td> used to initialize variables and values only when the case starts (t=0),
</td>
</tr>
<tr>
<td> **continue-init** </td>
<td> used to initialize structures specific to the restart mode. In principle,
a module should not have to perform specific operations in this case,</td>
</tr>
<tr>
<td> **build** </td>
<td> entrypoint called before initialization; the dataset has not yet been read.
This entrypoint is generally used to build certain objects useful to the module
but is rarely used by numerical modules.</td>
</tr>
<tr>
<td> **on-mesh-changed** </td>
<td> used to initialize variables and values during a change in the mesh
structure (partitioning, cell abandonment...). <strong>Attention</strong>: the
size of the code variables defined on mesh entities is automatically updated by
%Arcane.</td>
</tr>
<tr>
<td> **restore** </td>
<td> used to initialize specific structures during a rollback,</td>
</tr>
<tr>
<td> **exit** </td>
<td> entrypoint called at the end of execution. It is used for
example, used to deallocate data structures when the code exits: end
of simulation, stop before restart...</td>
</tr>
</table>

When the module descriptor is compiled by %Arcane (using **axl2cc** - see
previously), the entrypoints are registered within the architecture database.

Entrypoints must be declared at the module class level (otherwise an error
occurs during compilation):

```cpp
class TestModule
{
  ...

 public:

   void testPressureSync() override;
   void dumpConnection() override;
   ...
};
```

## Construction {#arcanedoc_core_types_axl_entrypoint_build}

Entrypoints are defined in the module definition file, in our case `TestModule.cc`.

For example, here is the `testPressureSync` entrypoint called at each iteration
of the computation loop. This entrypoint calculates the average of the cell
pressures over time:

```cpp
using namespace Arcane;

VariableNodeReal m_node_pressure = ...;
VariableCellReal m_cell_pressure = ...;

void TestModule::
testPressureSync()
{
  m_global_deltat = options()->deltatInit();
  m_node_pressure.fill(0.0);

  // Adds to each node the pressure of each cell it belongs to
  ENUMERATE_(Cell, i, allCells()){
    Cell cell = *i;
    Real cell_pressure = m_pressure[i];
    for( Node node : cell.nodes() )
      m_node_pressure[node] += pressure;
  }

  // Calculates the average pressure.
  ENUMERATE_(Node, i, allNodes()){
    Node node = *i;
    m_node_pressure[i] /= node.nbCell();
  }

  // Assigns to each cell the average pressure of the nodes that compose it
  ENUMERATE_(Cell, i, allCells()){
    Cell cell = *i;
    Integer nb_node = cell.nbNode();
    Real cell_pressure = 0.;
    for( Node node : cell.nodes())
      cell_pressure += m_node_pressure[node];
    cell_pressure /= nb_node;
    m_pressure[i] = cell_pressure;
  }

  // Synchronizes the pressure (for parallel execution)
  m_pressure.synchronize();

 // Calculates the minimum, maximum, and average pressure over all
 // domains
 Real min_pressure = 1.0e10;
 Real max_pressure = 0.0;
 Real sum_pressure = 0.0;

 ENUMERATE_(Cell, i, ownCells()){
   Real pressure = m_pressure[i];
   sum_pressure += pressure;
   if (pressure<min_pressure)
     min_pressure = pressure;
   if (pressure>max_pressure)
     max_pressure = pressure;
 }

 Real gmin = parallelMng()->reduce(Parallel::ReduceMin,min_pressure);
 Real gmax = parallelMng()->reduce(Parallel::ReduceMax,max_pressure);
 Real gsum = parallelMng()->reduce(Parallel::ReduceSum,sum_pressure);

 info() << "Local  Pressure: " << " Sum " << sum_pressure
        << " Min " << min_pressure << " Max " << max_pressure;
 info() << "Global Pressure: " << " Sum " << gsum
        << " Min " << gmin << " Max " << gmax;
}
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_variable
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_axl_caseoptions
</span>
</div>
