# Time Loop {#arcanedoc_core_types_timeloop}

[TOC]

A simulation code built with the %Arcane platform consists of a set of
\ref arcanedoc_core_types_module "numerical modules". These modules contain
\ref arcanedoc_core_types_axl_variable "variables" and
\ref arcanedoc_core_types_axl_entrypoint "entry points".

The sequence of calculations in the code is described by a succession of entry
points, the time loop. Time loops are defined in the application configuration
file (see \ref arcanedoc_core_types_codeconfig).

When running a case, the desired time loop for the case is chosen in the
dataset. Changing the time loop in the simulation dataset or modifying the time
loop description file does not require recompilation.

## Usage {#arcanedoc_core_types_timeloop_use}

To perform a simulation, a time loop must be chosen from those defined in the
previous configuration file. This selection is made in the *timeloop* element of
the *arcane* element in the dataset. The time loop is identified by its name.

```xml
<?xml version='1.0'?>
<case codeversion="1.0" codename="MicroHydro" xml:lang="en">
  <arcane>
    <title>Arcane Example of a very, very simplified Hydro module</title>
    <timeloop>MicroHydroLoop</timeloop>
  </arcane>
  ...
</case>
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_codeconfig
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_array_usage
</span>
</div>
