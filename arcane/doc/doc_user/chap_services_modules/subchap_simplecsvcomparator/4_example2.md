# Example No. 2 {#arcanedoc_services_modules_simplecsvcomparator_example2}

[TOC]

This Example 2 is identical to Example 1, without the use of singletons.


## .axl File -- Options Part

To start, here are the options for the axl file:

`SimpleTableComparatorExample2.axl`
\snippet SimpleTableComparatorExample2.axl SimpleTableComparatorExample2_options

We can see the presence of the two services.


## .arc File -- Module Option Part

Here is the corresponding .arc:

`SimpleTableComparatorExample2.arc`
\snippet SimpleTableComparatorExample2.arc SimpleTableComparatorExample2_arc

What we can see here is that the comparator has no options.


## Initial Entry Point

Here is the `start-init` entry point:

`SimpleTableComparatorExample2Module.cc`
\snippet SimpleTableComparatorExample2Module.cc SimpleTableComparatorExample2_init



## Loop Entry Point

The `compute-loop` entry point:

`SimpleTableComparatorExample2Module.cc`
\snippet SimpleTableComparatorExample2Module.cc SimpleTableComparatorExample2_loop



## Exit Entry Point

Finally, here is the `exit` entry point:

`SimpleTableComparatorExample2Module.cc`
\snippet SimpleTableComparatorExample2Module.cc SimpleTableComparatorExample2_exit

No surprises, it is the same usage.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_example1
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_example3
</span>
</div>
