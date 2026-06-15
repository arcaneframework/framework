# Examples: Generalities {#arcanedoc_services_modules_simplecsvcomparator_examples}

[TOC]

In the following chapters, some simple examples will be presented to you.

The 3 examples presented in this subsection are functional and are located in
the folder: `framework/arcane/samples_build/samples/simple_csv_comparator/`.

It should also be noted that these examples will work regardless of the
implementation of \arcane{ISimpleTableComparator} (just change the
implementation in `.config` (for singleton mode) or in the `.arc` files).

These examples share common structures: three entry points (`initModule`,
`loopModule`, `endModule`) representing three types of entry points
(`start-init`, `compute-loop`, `exit`) (in case:
\ref arcanedoc_core_types_axl_entrypoint) and no variables.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_usage
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_example1
</span>
</div>
