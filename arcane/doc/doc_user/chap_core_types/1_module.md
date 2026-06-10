# Module {#arcanedoc_core_types_module}

[TOC]

A module is a set of **entry points** and **variables**. It may possess
configuration options that allow the user to parameterize the module via the
simulation data set.

A module is represented by a class and an XML file called the
**module descriptor**.

## Module Descriptor {#arcanedoc_core_types_module_desc}

The module descriptor is an XML file with the ".axl" extension. It presents the
characteristics of the module:
- its variables,
- its entry points,
- its configuration options.

```xml
<?xml version="1.0"?>
<module name="Hydro" version="1.0">
  <name lang="fr">Hydro</name>

  <description>Hydro module descriptor</description>

  <variables>
  </variables>

  <entry-points>
  </entry-points>

  <options>
    <!-- Service of type IEquationOfState. -->
  </options>
</module>
```

For example, the file \c Hydro.axl above presents the module named **Hydro**,
whose base class is \c HydroModule (general case).
<strong>TODO: add reference doc on other module attributes.</strong>

The variables, entry points, and options will be described in chapter
\ref arcanedoc_core_types_axl.

## Class Representing the Module {#arcanedoc_core_types_module_class}

Thanks to the \c axl2cc utility, the \c Hydro.axl file generates a Hydro_axl.h
file. This file contains the \c ArcaneHydroObject class, the base class for the
Hydro module.

```cpp
class HydroModule
: public ArcaneHydroObject
{
 public:
  // Constructs a module with the parameters specified in \a mb
  ModuleHydro(const ModuleBuildInfo & mbi)
  : ArcaneHydroObject(mbi) {}

  // Returns the version number of the module
  virtual VersionInfo versionInfo() const { return VersionInfo(1, 0, 0); }
};
```

The previous example shows that %Arcane requires the module constructor to take
an object of type \c ModuleBuildInfo as a parameter to pass to its base class.
%Arcane also requires the definition of a \c versionInfo() method that returns
the version number of your module.

\note
Deriving from the ArcaneHydroObject class gives access, among other
things, to the %Arcane traces (see \ref arcanedoc_execution_traces) and the
following methods:
<table>
<tr><th>Method</th><th>Action</th></tr>
<tr><td>\c allCells() </td><td> returns the group of all cells </td></tr>
<tr><td>\c allNodes() </td><td> returns the group of all nodes </td></tr>
<tr><td>\c allFaces() </td><td> returns the group of all faces </td></tr>
<tr><td>\c ownCells() </td><td> returns the group of cells belonging to the subdomain </td></tr>
<tr><td>\c ownNodes() </td><td> returns the group of nodes belonging to the subdomain </td></tr>
<tr><td>\c ownFaces() </td><td> returns the group of all faces belonging to the subdomain </td></tr>
</table>

## Connecting the Module to Arcane {#arcanedoc_core_types_module_connectarcane}

An instance of the module is constructed by the architecture during execution.

The user must therefore provide a function to create an instance of the module
class. %Arcane provides a macro to define a generic creation function. This
macro must be written in the source file where the module is defined. It has the
following prototype:

```cpp
ARCANE_REGISTER_MODULE_HYDRO(HydroModule);
```

\c *HydroModule* corresponds to the class name and *HYDRO* following
**ARCANE_REGISTER_MODULE_** allows defining the creation function.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_service
</span>
</div>
