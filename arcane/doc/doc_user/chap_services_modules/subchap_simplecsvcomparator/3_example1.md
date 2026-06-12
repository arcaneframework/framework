# Example No. 1 {#arcanedoc_services_modules_simplecsvcomparator_example1}

[TOC]

In this first example, we will reuse example 1 from the previous subsection
(\ref arcanedoc_services_modules_simplecsvoutput_example1) but by modifying the
`exit` entry point.

We will use both services as singletons here.

## .config File

To start, let's look at the `.config`:

`csv.config` `<time-loop name="example1">`
\snippet stc.config SimpleTableComparatorExample1_config

Here, we see the two singletons that we will use.

## Initial Entry Point

Here is the `start-init` entry point:

`SimpleTableComparatorExample1Module.cc`
\snippet SimpleTableComparatorExample1Module.cc SimpleTableComparatorExample1_init


## Loop Entry Point

The `compute-loop` entry point:

`SimpleTableComparatorExample1Module.cc`
\snippet SimpleTableComparatorExample1Module.cc SimpleTableComparatorExample1_loop


## Exit Entry Point

Finally, let's look at the `exit` entry point:

`SimpleTableComparatorExample1Module.cc`
\snippet SimpleTableComparatorExample1Module.cc SimpleTableComparatorExample1_exit

We can see a minimal example of using the comparator.
We start by retrieving the pointer to the singleton, then we initialize the
comparator by giving it a pointer to an object implementing the
\arcane{ISimpleTableOutput} interface.

Next, we check if a reference file exists.
If there isn't one, we create it using the values from `table`. Indeed, the
`SimpleCsvComparator` service is also capable of writing files. It will use the
information from \arcane{ISimpleTableOutput} to find the path.

If we look at the `init` entry point, we can see the initialization of the
`SimpleCsvOutput` service:

```cpp
table->init("Results_Example1", "example1");
```

`SimpleCsvComparator` will use this information to write the reference file. In
this example, it will write the file here:
```sh
./output/csv_refs/example1/Results_Example1.csv
```

If the reference file already exists, the comparator will compare it with the
values of the `table` object.
If the values are identical, we will have the message `Mêmes valeurs !!!` in the
output; otherwise, we will have `Valeurs différentes :(` (and an error code
`1`).

Finally, the `SimpleCsvOutput` service writes its file, as usual.

\remark
As you have understood, you must run the example twice to see what happens.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_examples
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_example2
</span>
</div>
