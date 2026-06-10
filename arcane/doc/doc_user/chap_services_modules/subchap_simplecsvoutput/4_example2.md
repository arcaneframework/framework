# Example No. 2 {#arcanedoc_services_modules_simplecsvoutput_example2}

[TOC]

Example 2 is almost identical to Example 1, apart from the definition of the
table name (and file) and the subdirectory name.

As a **singleton**, our service does not have access to the dataset, so we must
go through one of the modules to transfer the information.

Example 2 shows how to do this simply.

## .axl File -- <options> Section

To start, let's look at the options of the axl file:

`SimpleTableOutputExample2.axl`
\snippet SimpleTableOutputExample2.axl SimpleTableOutputExample2_options

If you take a look at the service's `.axl` (here:
\ref axldoc_service_SimpleCsvOutput_arcane_std), you will notice that it is
identical.
Indeed, in the case of the singleton, we use our main module to retrieve the
information our service needs.

Another thing to note: the default value `default=""`.
Setting an empty default value allows us to determine whether the user specified
a value in the `.arc` or not.
Later, in the module, we could say that if there are no values for both options,
then the user simply does not want CSV output.
(this is the method used in QAMA).

## .arc File -- Module Option Section

Here is the corresponding `.arc`:

`SimpleTableOutputExample2.arc`
\snippet SimpleTableOutputExample2.arc SimpleTableOutputExample2_arc

As explained above, the module manages the two options, so we are in the "module
option" section.


## Initial Entry Point

Let's look at the `start-init` entry point:

`SimpleTableOutputExample2Module.cc`
\snippet SimpleTableOutputExample2Module.cc SimpleTableOutputExample2_init

This is the module that manages the service's two options, including the default
values. If the user does not specify a value for the `tableName` option in the
`.arc`, we define a default name (knowing that the service also does this if
`table->init()` is called without parameters).

\warning
A call to init like this: `table->init()` is different from a call to init like
this: `table->init("")`! One will take a default value, the other will have an
empty name, and the output file will simply have no name (just the extension).


## Loop Entry Point

This entry point is identical to that of Example 1.


## Exit Entry Point

Finally, let's look at the `exit` entry point:

`SimpleTableOutputExample2Module.cc`
\snippet SimpleTableOutputExample2Module.cc SimpleTableOutputExample2_exit


The line
```cpp
if(options()->getTableName() != "" || options()->getTableDir() != "")
```
allows us to know if the user entered at least one of the options.
If so, we check the default value and write the file.
If not, we do not write a file.




____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example1
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example3
</span>
</div>
