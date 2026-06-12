# Module/Service Descriptor (.AXL) {#arcanedoc_core_types_axl}

As previously explained, the module/service descriptor is a file
with the `.axl` extension that accompanies modules and services and describes
the following elements:
- where the interfaces it implements (**service only**),
- its variables,
- its entry points (**module only**),
- its configuration options.

This subsection will therefore go into detail and explain these three
essential principles in %Arcane.

As a reminder, a module descriptor looks like this:

```xml
<?xml version="1.0"?>
<module name="Hydro" version="1.0">
  <description>Hydro module descriptor</description>

  <variables>
    <!-- See "Variable" section. -->
  </variables>
  <entry-points>
    <!-- See "Entry Point" section. -->
  </entry-points>
  <options>
    <!-- Service of type IEquationOfState. -->
    <!-- See "Options" section. -->
  </options>
</module>
```

And a service descriptor looks like this:

```xml
<?xml version="1.0"?>
<service name="PerfectGasEOS" version="1.0">
  <description>PerfectGasEOSService descriptor</description>

  <interface name="IEquationOfState"/>

  <variables>
    <!-- See "Variable" section. -->
  </variables>
  <options>
    <!-- See "Options" section. -->
  </options>
</service>
```

<br>

Table of Contents for this subsection:

1. \subpage arcanedoc_core_types_axl_variable <br>
   Presents the concept of variables in %Arcane.

2. \subpage arcanedoc_core_types_axl_entrypoint <br>
   Presents the concept of entry points in %Arcane.

3. \subpage arcanedoc_core_types_axl_caseoptions <br>
   Explains how to configure modules with user options provided in the dataset.

4. \subpage arcanedoc_core_types_axl_others <br>
   Presents details that do not appear in other sub-chapters.

____

<div class="section_buttons">
<span class="next_section_button">
\ref arcanedoc_core_types_axl_variable
</span>
</div>
