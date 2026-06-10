# SimpleCsvComparator Service {#arcanedoc_services_modules_simplecsvcomparator}

[TOC]

- Generated documentation for the service is available here:
  \ref axldoc_service_SimpleCsvComparator_arcane_std
- Documentation for the interface implemented by this service:
  \arcane{ISimpleTableComparator}

\warning
The interface is not yet finalized. It may still evolve.

____

\warning
This subsection was designed as a follow-up to the subsection
\ref arcanedoc_services_modules_simplecsvoutput.

This service allows comparing the values of two `SimpleTableInternal` objects
against each other.
During a code run that integrates a service of type \arcane{ISimpleTableOutput},
it is possible to generate a reference file (or several, one per subdomain, if
desired).

Then, during a subsequent run, it is possible to compare the values from the
reference file generated previously with the values stored in the service of
type \arcane{ISimpleTableOutput} of the current run.

Thanks to the CSV format, it is also possible to view and modify the reference
values, if desired.

This service can be used as a standard service defined in a module's AXL or as
a singleton to have a unique instance for all modules.

This subsection introduces this service. Not all use cases will be covered,
so it is recommended to consult the documentation for the
\arcane{ISimpleTableComparator} interface to fully utilize this service.

<br>

Table of Contents for this subsection:

1. \subpage arcanedoc_services_modules_simplecsvcomparator_usage <br>
   Summarizes how to use the service.

2. \subpage arcanedoc_services_modules_simplecsvcomparator_examples <br>
   Some general information to read before tackling the examples.

3. \subpage arcanedoc_services_modules_simplecsvcomparator_example1 <br>
   This simple example introduces how to use the service in singleton mode.

4. \subpage arcanedoc_services_modules_simplecsvcomparator_example2 <br>
   This example does not use singleton mode.

5. \subpage arcanedoc_services_modules_simplecsvcomparator_example3 <br>
   This example mixes a singleton `SimpleCsvOutput` and a non-singleton
   `SimpleCsvComparator`. There is also an example of using regular expressions.

____


<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_usage
</span>
</div>
