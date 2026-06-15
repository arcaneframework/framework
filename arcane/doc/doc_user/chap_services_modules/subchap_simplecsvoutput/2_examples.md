# Examples: generalities {#arcanedoc_services_modules_simplecsvoutput_examples}

[TOC]

In the following chapters, some simple examples will be presented to you.

When describing tables, to highlight the number of rows and columns, the row and
column containing the headers are omitted from the counts, just as in the
service.

The 6 examples presented in this subsection are functional and are located in
the folder:
`framework/arcane/samples_build/samples/simple_csv_output/`.

It should also be noted that these examples will work regardless of the
implementation of \arcane{ISimpleTableOutput} (simply change the implementation
in `.config` (for singleton mode) or in the `.arc` files).

These examples share common structures: three entry points (`initModule`,
`loopModule`, `endModule`) representing three types of entry points
(`start-init`, `compute-loop`, `exit`) (in case:
\ref arcanedoc_core_types_axl_entrypoint) and no variables.
The options vary for examples 1, 2, and 3-6.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_usage
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example1
</span>
</div>
