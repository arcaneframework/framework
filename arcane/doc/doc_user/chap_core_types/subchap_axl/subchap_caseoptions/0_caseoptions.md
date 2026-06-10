# Options {#arcanedoc_core_types_axl_caseoptions}

This subsection describes the possible options for the *axl* file. These options
apply identically to modules and services. To avoid unnecessary repetition, we
will use the term module only, knowing that this also applies to services.

Each module has options that can be specified by the user when launching an
execution. These options are generally dictated by the *dataset* that the user
provides to run their case. The document \ref arcanedoc_core_types_module shows
that each module has a configuration file named *module descriptor* consisting
of 3 parts: variables, entry points, and configuration options. This document
focuses on the part concerning configuration options, which will allow defining
the grammar of the module's dataset.

The module descriptor is an XML file. This file is used by %Arcane to generate
C++ classes. One of these classes is responsible for reading the information in
the dataset.

By convention, for a module called *Test*, the module descriptor is named
*Test.axl*. This `axl` file allows the generation of a *Test_axl.h* file. This
file will be included by the class implementing the module.

In the dataset, a module's options are provided in the `<options>` element. For
example, the options for the *Test* module are:

```xml
<module name="Test" version="1.0">
  <name lang="fr">Test</name>
  <description>Module Test</description>
  <options>
    <!-- contient les options du module Test -->
    ...
  </options>
</module>
```

<br>

Table of Contents for this subsection:

1. \subpage arcanedoc_core_types_axl_caseoptions_struct <br>
   Presents the structure of an option in an AXL file.

2. \subpage arcanedoc_core_types_axl_caseoptions_common_struct <br>
   Presents the attributes and properties common to options.

3. \subpage arcanedoc_core_types_axl_caseoptions_options <br>
   Presents all types of options that can be defined in an AXL file.

4. \subpage arcanedoc_core_types_axl_caseoptions_usage <br>
   Presents how to use an option in a module.

5. \subpage arcanedoc_core_types_axl_caseoptions_default_values <br>
   Presents how to manage default values for options in case the ARC file does
   not define any values.




____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_entrypoint
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_axl_caseoptions_struct
</span>
</div>
