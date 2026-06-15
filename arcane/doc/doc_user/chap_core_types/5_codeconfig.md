# Code Configuration (.CONFIG) {#arcanedoc_core_types_codeconfig}

[TOC]

## Introduction {#arcanedoc_core_types_codeconfig_intro}

<!-- presents the configuration file of an executable -->
<!-- created with the %Arcane platform. This file contains, among other things,
the description of the available time loops. -->

Code configuration is described in an external file, named CODE.config, where
\a CODE is the name of the code.

This file describes all the time loops (see \ref arcanedoc_core_types_timeloop)
available for the code, as well as their configuration.

## File Structure {#arcanedoc_core_types_codeconfig_struct}

This application configuration file is in XML format.
Here is an example of such a file for a <em>MicroHydro</em> module located in
the ARCANE examples directory (\c samples):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<arcane-config code-name="MicroHydro">
  <time-loops>
    <time-loop name="MicroHydroLoop">
      <title>MicroHydro</title>
      <description>Time loop of the Arcane MicroHydro example</description>

      <modules>
        <module name="MicroHydro" need="required"/>
        <module name="ArcanePostProcessing" need="required"/>
      </modules>

      <singleton-services>
      </singleton-services>

      <entry-points where="init">
        <entry-point name="MicroHydro.HydroStartInit"/>
      </entry-points>

      <entry-points where="compute-loop">
        <entry-point name="MicroHydro.ComputePressureForce"/>
        <entry-point name="MicroHydro.ComputeVelocity"/>
        <entry-point name="MicroHydro.ApplyBoundaryCondition"/>
        <entry-point name="MicroHydro.MoveNodes"/>
        <entry-point name="MicroHydro.ComputeGeometricValues"/>
        <entry-point name="MicroHydro.UpdateDensity"/>
        <entry-point name="MicroHydro.ApplyEquationOfState"/>
        <entry-point name="MicroHydro.ComputeDeltaT"/>
      </entry-points>

    </time-loop>
  </time-loops>
</arcane-config>
```

## The <time-loops> Element {#arcanedoc_core_types_codeconfig_timeloop}

The set of time loops is described in the `<time-loops>` element. Each time loop
is represented by the `<time-loop>` element and identified by its name (the
`name` attribute). The previous file therefore describes a single time loop
named <em>MicroHydroLoop</em>.

In addition to the loop's title and description, there are 3 elements:
`<modules>`, `<singleton-services>`, and `<entry-points>`.

### The <modules> Element {#arcanedoc_core_types_codeconfig_modules}

This element describes all the modules of the code necessary for the execution
of the time loop. The `name` attribute identifies the module by its name, and
the `need` attribute (valued **required** or **optional**) indicates whether the
module must be present or not. If the module is not mandatory and is not
provided during execution (absence of the module library), its entry points will
be ignored. This allows for the construction of variants of the same time loop.

### The <singleton-services> Element {#arcanedoc_core_types_codeconfig_singletonservices}

This element is quite similar to `<modules>` and describes all the singleton
services of the code used during execution. A singleton service is a service for
which only one instance exists, created during the code's initialization. Such a
service may have options in the dataset. The specification of singleton services
is done as follows:

```xml
<singleton-services>
  <service name="Toto" need="required">
  <service name="Tutu" need="optional">
</singleton-services>
```

Like the module, there are two attributes, `name` and `need`. If the service is
optional and not found, it will not be instantiated. In the code, it is possible
to retrieve a singleton service whose interface is known, via the ServiceBuilder
class. Example:

```cpp
m_toto = ServiceBuilder<IToto>(subDomain()).getSingleton();
```

With Toto implementing the IToto interface.

### The <entry-points> Element {#arcanedoc_core_types_codeconfig_entrypoints}

This element contains the `where` attribute specifying the location where the
different entry points are called. The possible values are described in the
table below:

<table>

<tr>
<th>Value</th>
<th>Description</th>
</tr>

<tr>
<td> **build** </td>
<td>Entry point called when the module is created. When the module is created,
the mesh has not yet been loaded and therefore should not be used.</td>
</tr>

<tr>
<td> **init** </td>
<td>Entry point called during the code's initialization.</td>
</tr>

<tr>
<td> **compute-loop** </td>
<td>Entry point called in the iteration loop.</td>
</tr>

<tr>
<td> **restore** </td>
<td>Entry point called during a rollback.</td>
</tr>

<tr>
<td> **on-mesh-changed** </td>
<td>Entry point called when the mesh structure changes (partitioning, cell
abandonment...).</td>
</tr>

<tr>
<td> **exit** </td>
<td>Entry point called before the code exits: end of simulation, stop before
restart...</td>
</tr>

</table>

The `<entry-points>` element contains a list of entry points named
*module_name.entry_point_name*. The value of the `where` attribute of each entry
point (defined in \ref arcanedoc_core_types_module_desc "the module descriptor")
must be compatible with the value of the `where` attribute of the
`<entry-points>` block. In an *init* block, only entry points whose `where`
attribute equals **init**, **start-init**, or **continue-init** can be present.
For other blocks (**compute-loop**, **restore**, **on-mesh-changed**, **exit**),
the values of the two `where` attributes must be identical.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_casefile
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_timeloop
</span>
</div>
