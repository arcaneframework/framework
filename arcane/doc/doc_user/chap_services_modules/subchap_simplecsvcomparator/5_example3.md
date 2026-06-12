# Example No. 3 {#arcanedoc_services_modules_simplecsvcomparator_example3}

[TOC]

With Example 3, we mixed singleton and normal. The `SimpleCsvOutput` service is
a singleton, but `SimpleCsvComparator` is not.

## Initial Entry Point

`start-init` Entry Point:

`SimpleTableComparatorExample3Module.cc`
\snippet SimpleTableComparatorExample3Module.cc SimpleTableComparatorExample3_init



## Loop Entry Point

`compute-loop` Entry Point:

`SimpleTableComparatorExample3Module.cc`
\snippet SimpleTableComparatorExample3Module.cc SimpleTableComparatorExample3_loop


## Exit Entry Point

Let's look at the `exit` entry point:

`SimpleTableComparatorExample3Module.cc`
\snippet SimpleTableComparatorExample3Module.cc SimpleTableComparatorExample3_exit

Here, we added three things: two `editElement()` calls and one
`editRegexRows()`. The `editElement()` calls allow modifying an element to cause
an error during comparison. The `editRegexRows()` allows choosing the rows we
want to compare.

In this case, only rows containing `Fissions` in their name will be compared (so
here, only the `Nb de Fissions` row).

Indeed, you can choose which rows and columns you want to compare. You can do
this using regular expressions or by specifying their name directly (via the
methods \arcane{ISimpleTableComparator::addRowForComparing()} and
\arcane{ISimpleTableComparator::addColumnForComparing()}).

It is also possible to specify that the rows/columns you provide are
rows/columns you want to exclude from comparison (and not include) (methods
\arcane{ISimpleTableComparator::isAnArrayExclusiveRows()},
\arcane{ISimpleTableComparator::isAnArrayExclusiveColumns()},
\arcane{ISimpleTableComparator::isARegexExclusiveRows()} and
\arcane{ISimpleTableComparator::isARegexExclusiveColumns()}).

Finally, it is possible to specify an epsilon to have an acceptable margin of
error (\arcane{ISimpleTableComparator::addEpsilonRow()} /
\arcane{ISimpleTableComparator::addEpsilonColumn()}).

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_example2
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_example4
</span> -->
</div>
