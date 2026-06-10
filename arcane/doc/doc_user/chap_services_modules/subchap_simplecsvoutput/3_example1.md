# Example No. 1 {#arcanedoc_services_modules_simplecsvoutput_example1}

[TOC]

Let's start with Example 1.
This simple example allows outputting a table that looks like this:

| Results_Example1 | Iteration 1 | Iteration 2 | Iteration 3 |
|------------------|-------------|-------------|-------------|
| Nb de Fissions   | 36          | 0           | 85          |
| Nb de Collisions | 29          | 84          | 21          |

We have two rows and three columns here.
The numbers present are generated randomly.

## Initial entry point

Let's look at the `start-init` entry point:

`SimpleTableOutputExample1Module.cc`
\snippet SimpleTableOutputExample1Module.cc SimpleTableOutputExample1_init

In this example, we are in singleton mode. The `.axl` file for this example
therefore does not contain options.

However, the `.config` configuration file references the singleton.

`csv.config` `<time-loop name="example1">`
\snippet csv.config SimpleTableOutputExample1_config

\note
To be able to separate several execution cases in the same code, I created
several `time-loop`s in the `.config` file.

Let's go back to the `.cc` file. We only retrieve the pointer to the singleton
and initialize it with a table name (`Results_Example1`) and a subdirectory
name (`example1`) (and we print the empty table).


## Loop entry point

Let's look at the `compute-loop` entry point:

`SimpleTableOutputExample1Module.cc`
\snippet SimpleTableOutputExample1Module.cc SimpleTableOutputExample1_loop

Here, we have a simple example of the typical usage imagined initially for this
service.

We create a column named `Iteration X` (meaning a new column for each
iteration), then we add the desired values to rows.

Do the rows not exist because they were not created during init? The service
takes care of creating them before adding the value.

In this example, the rows will therefore be created during the first iteration,
and afterwards, the service will simply fill them.

\note
We can interchange `row` and `column`; the functionality is identical for rows
and columns (but the result will obviously be different).

Exporting values from code, for example during a debugging session, is therefore
something that is very simple with this service.


## Exit entry point

Finally, let's look at the `exit` entry point:

`SimpleTableOutputExample1Module.cc`
\snippet SimpleTableOutputExample1Module.cc SimpleTableOutputExample1_exit

In this part, we simply request the writing of the CSV file whose location and
name were defined during init. If this had not been done, there are the methods
\arcane{ISimpleTableOutput::setOutputDirectory()} and
\arcane{ISimpleTableOutput::setTableName()} to correct it (otherwise, the
service defines default values).




____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_examples
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example2
</span>
</div>
