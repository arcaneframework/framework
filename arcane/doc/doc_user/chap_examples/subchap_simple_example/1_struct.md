# General Structure {#arcanedoc_examples_simple_example_struct}

[TOC]

## HelloWorld {#arcanedoc_examples_simple_example_struct_helloworld}

Here is a diagram representing the structure of our Hello World:

\image html HW_schema.svg

In this application, we have a module called `SayHello` containing three files:
- a header (.h),
- a source file (.cc),
- a file containing the dataset options (.axl).

And outside the module, we have four files:
- a `main.cc` file allowing us to run our application,
- a `CMakeLists.txt` file allowing us to compile our application,
- a `.config` file allowing us to configure our application,
- a `.arc` file containing a dataset for our application.

All these elements constitute our `HelloWorld` application.

\note
It is possible to generate an application template using the
`arcane-template` program.
To generate a template for our `HelloWorld` using `arcane-template`, here is the
command:
```sh
./arcane_templates generate-application -code-name HelloWorld --module-name SayHello --output-directory ~/HelloWorld
```
This program can be found in the `bin` folder of the %Arcane installation
directory: `arcane_install/bin/`.


In the following section, we will look at the `SayHello` module.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_simple_example
</span>
<span class="next_section_button">
\ref arcanedoc_examples_simple_example_module
</span>
</div>
