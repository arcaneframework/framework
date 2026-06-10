# Service SimpleCsvOutput {#arcanedoc_services_modules_simplecsvoutput}

[TOC]

- Generated service documentation is available here:
  \ref axldoc_service_SimpleCsvOutput_arcane_std
- Documentation of the interface implemented by this service:
  \arcane{ISimpleTableOutput}

\warning
The interface is not yet finalized. It may still evolve.

____

This service allows you to create a 2D table of values with named rows and
columns.
Currently, the output file format is CSV format.
This service can be used as a standard service defined in a module's AXL or as a
singleton to have a unique instance for all modules.

You just need to create one or more rows and one or more columns, then assign
values to each [row,column], and finally call the writeFile() method to generate
a .csv file.

Example .csv file:
```csv
Results_Example3;Iteration 1;Iteration 2;Iteration 3;
Nb de Fissions;36;0;85;
Nb de Collisions;29;84;21;
```
In Excel (or another spreadsheet program), you get this table:
| Results_Example3 | Iteration 1 | Iteration 2 | Iteration 3 |
|------------------|-------------|-------------|-------------|
| Nb de Fissions   | 36          | 0           | 85          |
| Nb de Collisions | 29          | 84          | 21          |

This subsection introduces this service. Not all use cases will be covered, so
it is recommended to view the documentation for the \arcane{ISimpleTableOutput}
interface to fully exploit this service (notably the multi-process management
aspect, which does not have an example).

<br>

Table of Contents for this subsection:

1. \subpage arcanedoc_services_modules_simplecsvoutput_usage <br>
   Summarizes how to use the service (as a singleton, in parallel).

2. \subpage arcanedoc_services_modules_simplecsvoutput_examples <br>
   Some general information to read before tackling the examples.

3. \subpage arcanedoc_services_modules_simplecsvoutput_example1 <br>
   This simple example introduces how to use the service in singleton mode.

4. \subpage arcanedoc_services_modules_simplecsvoutput_example2 <br>
   This example also uses singleton mode and shows how to pass options to the
   service through the main module.

5. \subpage arcanedoc_services_modules_simplecsvoutput_example3 <br>
   This example is the same as example 1 but without singleton mode.

6. \subpage arcanedoc_services_modules_simplecsvoutput_example4 <br>
   This example explains how to use the service in a more optimal way.

7. \subpage arcanedoc_services_modules_simplecsvoutput_example5 <br>
   This example introduces filling the table by using the cell pointer and
   directional movements.

8. \subpage arcanedoc_services_modules_simplecsvoutput_example6 <br>
   This example continues the explanation of filling using the cell pointer.

____


<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_usage
</span>
</div>
