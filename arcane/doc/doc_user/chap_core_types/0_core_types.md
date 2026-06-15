# Fundamental Types {#arcanedoc_core_types}

There are 4 fundamental types in %Arcane, which correspond to the concepts of
**Module**, **Service**, **Variable**, and **Entry Point**.
These 4 fundamental types are present in three types of files specific to
%Arcane: **AXL File** (Module/Service Descriptor), **ARC File** (Data Set), and
**Config File**.

For a brief description of these concepts, refer to the chapter
\ref arcanedoc_getting_started.

Here is a simple %Arcane code schema, with one module and two services. The two
services share a common interface.

\image html code_schema.svg

The different parts of this chapter should allow you to understand this schema
(apart from the main.cc and CMakeLists.txt files, which are explained in chapter
\ref arcanedoc_execution).

<br>

Table of Contents for this chapter:
1. \subpage arcanedoc_core_types_module <br>
  Presents the concept of a module in %Arcane.

2. \subpage arcanedoc_core_types_service <br>
  Presents the concept of a service in %Arcane.

3. \subpage arcanedoc_core_types_axl <br>
  Presents everything you need to know about module/service descriptors
  (represented by files with the .axl extension). This subsection presents the
  concepts of \ref arcanedoc_core_types_axl_variable and
  \ref arcanedoc_core_types_axl_entrypoint.

4. \subpage arcanedoc_core_types_casefile <br>
  Presents the syntax of the data set (represented by files with the .arc
  extension).

5. \subpage arcanedoc_core_types_codeconfig <br>
  Presents the global code configuration file.

6. \subpage arcanedoc_core_types_timeloop <br>
  Describes the concept of a time loop.

7. \subpage arcanedoc_core_types_array_usage <br>
  Describes the use of array types.

8. \subpage arcanedoc_core_types_numarray <br>
  Describes the use of the NumArray class and associated types.

____

<div class="section_buttons">
<span class="next_section_button">
\ref arcanedoc_core_types_module
</span>
</div>
