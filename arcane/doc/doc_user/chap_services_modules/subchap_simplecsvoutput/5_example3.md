# Example No. 3 {#arcanedoc_services_modules_simplecsvoutput_example3}

[TOC]

With Example 3 and subsequent examples, we no longer use a singleton.
So we will see a simple example of using the service normally.

Note that this example does the same thing as the previous examples.


## .axl File -- <options> Section

To start, let's look at the options of the `.axl` file:

`SimpleTableOutputExample3.axl`
\snippet SimpleTableOutputExample3.axl SimpleTableOutputExample3_options

In the `.axl`, we just declare the use of a service implementing the
Arcane::ISimpleTableOutput interface.

## .arc File -- Module Option Section

Here is the corresponding `.arc`:

`SimpleTableOutputExample3.arc`
\snippet SimpleTableOutputExample3.arc SimpleTableOutputExample3_arc

Here, compared to the previous example, we fill in the options in the service
section.
We request the use of the `SimpleCsvOutput` service with the two options it
requires.


## Initial Entry Point

Let's look at the `start-init` entry point:

`SimpleTableOutputExample3Module.cc`
\snippet SimpleTableOutputExample3Module.cc SimpleTableOutputExample3_init

Compared to the previous example, we do not need to retrieve a pointer to a
singleton; here it is a service used normally.

Still compared to the previous example, the service manages the default values.

\note
For now, it is impossible to request the non-writing of output files directly in
the `.arc` service section. If you do not put values in the `tableDir` and
`tableName` options, writing will still occur. This must be handled by the
module for the time being.


## Loop Entry Point

Let's look at the `compute-loop` entry point:

`SimpleTableOutputExample3Module.cc`
\snippet SimpleTableOutputExample3Module.cc SimpleTableOutputExample3_loop

Aside from replacing the singleton pointer with the use of module options, there
are no differences from the two previous examples.


## Exit Entry Point

Finally, let's look at the `exit` entry point:

`SimpleTableOutputExample3Module.cc`
\snippet SimpleTableOutputExample3Module.cc SimpleTableOutputExample3_exit

Here, we simply write the output file (and print the table).
If one wishes to control whether or not the file is written via the `.arc`, this
is where it can be done by conditioning the call to `writeFile()` with an
`if()`.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example2
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example4
</span>
</div>
