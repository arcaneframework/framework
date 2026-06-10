# Usage {#arcanedoc_services_modules_simplecsvoutput_usage}

[TOC]

## Singleton

For use as a singleton (the same object for all modules):

Place these lines in your project's .config:

```xml
<singleton-services>
  <service name="SimpleCsvOutput" need="required" />
</singleton-services>
```

And in your module(s):

```cpp
#include <arcane/core/ISimpleTableOutput.h>

using namespace Arcane;

ISimpleTableOutput* table = ServiceBuilder<ISimpleTableOutput>(subDomain()).getSingleton();
table->init("Example_Name", "example"); // Must only be done by a single module.
// Using the service...
table->writeFile(); // Must only be done by a single module (unless you know what you are doing).
```

## Service

For use as a service (a different object for each module): 

Place these lines in your module's .axl:

```xml
<!-- <options> -->
  <service-instance name="simple-table-output" type="Arcane::ISimpleTableOutput">
    <description>Service implementing ISimpleTableOutput</description>
  </service-instance>
<!-- </options> -->
```

In the .arc, you can configure the service options. For example:

```xml
<!-- <mon-module> -->
  <simple-table-output name="SimpleCsvOutput">
    <!-- The name of the directory to create/use. -->
    <tableDir>example_dir</tableDir>
    <!-- The name of the file to create/overwrite. -->
    <tableName>Results_Example</tableName>

    <!-- Finally, we will have a file with the path: 
    ./output/csv/example_dir/Results_Example.csv -->
  </simple-table-output>
<!-- </mon-module> -->
```

And in your module:

```cpp
#include <arcane/ISimpleTableOutput.h>

using namespace Arcane;

options()->simpleCsvOutput()->init();
//...
options()->simpleCsvOutput()->writeFile();
```

You can also use the service in both ways at the same time, depending on your
needs.

(For a more concrete example, see the following pages)


## Naming Symbols for Parallel Execution (CSV Implementation)

In the directory name or the table name, whether in singleton mode or service
mode, it is possible to add *symbols* that will be replaced during execution.

The available *symbols* are:
- `@proc_id@`: Will be replaced by the process rank.
- `@num_procs@`: Will be replaced by the total number of processes.

For example, if we have:

```xml
<tableDir>N_@num_procs@</tableDir>
<tableName>Results_P@proc_id@</tableName>
```

or when initializing the service:

```cpp
...
table->init("Results_P@proc_id@", "N_@num_procs@");
...
```

And if we run the program with 2 processes (ID = 0 and 1), we will get two CSV
files with the path:
- `./output/csv/N_2/Results_P0.csv`
- `./output/csv/N_2/Results_P1.csv`

(sequentially, we will have `./output/csv/N_1/Results_P0.csv`)

This allows, among other things, to:
- create a table per process and name them easily,
- create "generic" .arc files where the number of processes does not matter,
- have a different name for each table, in the case where a *cat* is performed
  (reminder: *tableName* gives the name of the csv file but is also placed in
  the first cell of the table).



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_examples
</span>
</div>
