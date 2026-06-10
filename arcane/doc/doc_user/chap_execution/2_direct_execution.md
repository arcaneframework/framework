# Direct Execution Launch {#arcanedoc_execution_direct_execution}

[TOC]

It is possible to use %Arcane without going through the mechanisms that use
modules and the time loop. This can be useful for very simple codes or
utilities, but this mechanism is not recommended for large calculation codes
because it does not automatically allow access to all of %Arcane's features,
such as load balancing, safeguards/recoveries, or modules (even if these
mechanisms remain manually accessible).

There are two ways to launch standalone mode:

- the mode with accelerator support. In this mode, only the accelerator API and
  %Arcane's utility classes are available. The page \ref
  arcanedoc_parallel_accelerator_standalone describes how to use this mode.
- the subdomain mode. This mode allows manual access to most of %Arcane's
  features, such as meshing, parsing, or load balancing.

The two examples `standalone_subdomain` and `standalone_accelerator` show how to
use these mechanisms.

The page \ref arcanedoc_execution_launcher explains how to provide the
parameters to initialize %Arcane.

## Standalone Subdomain Mode {#arcanedoc_parallel_accelerator_standalone_subdomain}

This mode allows manual control of most of %Arcane's features, such as meshing
and parsing. To use this mode, simply use the class method
\arcane{ArcaneLauncher::createStandaloneSubDomain()} after initializing %Arcane:

```cpp
Arcane::String case_file_name = {};
Arcane::ArcaneLauncher::init(Arcane::CommandLineArguments(&argc, &argv));
Arcane::StandaloneSubDomain sub_domain(Arcane::ArcaneLauncher::createStandaloneSubDomain(case_file_name));
```

It is possible to specify a filename for the dataset. In this case, if this file
contains meshes, they will be automatically created when the subdomain is
created.

The `sub_domain` instance must remain valid as long as you wish to use the
subdomain. It is therefore preferable to define it in the code's `main()`.

\warning Only one call to \arcane{ArcaneLauncher::createStandaloneSubDomain} is
allowed.

For example, the following code reads a mesh, displays the number of meshes,
calculates, and displays the coordinates of the mesh centers.

\snippet standalone_subdomain/main.cc StandaloneSubDomainFull

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_execution_launcher
</span>
<span class="next_section_button">
\ref arcanedoc_execution_env_variables
</span>
</div>
