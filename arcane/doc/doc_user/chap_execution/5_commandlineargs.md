# Command Line and Dataset {#arcanedoc_execution_commandlineargs}

[TOC]

## Introduction {#arcanedoc_execution_commandlineargs_intro}

There are two ways to customize the dataset options via command line arguments:
- by symbol replacement,
- by option address *(recommended method)*.

It is possible to customize all types of %Arcane options listed here:
\ref arcanedoc_core_types_axl_caseoptions_options

## Customization by Symbols {#arcanedoc_execution_commandlineargs_symbol}

### Symbols in the Dataset {#arcanedoc_execution_commandlineargs_symbol_dataset}

This type of customization requires modifying the dataset to include symbols.
This dataset can therefore become unusable without the correct command line
arguments.

A symbol is a string of characters enclosed in at-signs.

Example: `@UneValeur@`

If we put this symbol in a dataset, we could have, for example:

```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>

  <simple-hydro>
    <deltat-cp>@UneValeur@</deltat-cp>
  </simple-hydro>

</case>
```

Whether the option `deltat-cp` is a *simple option*, an *enumerated option*, or
an *extended option*, the functionality is the same, even if these options are
in *complex options* or *service options*.

For *service options*, symbol replacement also works in the `name` and
`mesh-name` attributes. Example:

```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>

  <simple-hydro>
    <post-processor2 name="@NamePostProcessor@" mesh-name="@MeshPostProcessor@">
      <fileset-size>@NbTimeInOneFile@</fileset-size>
      <binary-file>false</binary-file>
    </post-processor2>
  </simple-hydro>

</case>
```

Three symbols can be replaced: `@NamePostProcessor@`, `@MeshPostProcessor@`, and
`@NbTimeInOneFile@`.

\remark Here, we can see a first limitation to symbol replacement for
*service options*: if we replace the symbol `@NamePostProcessor@` with
`Ensight7PostProcessor`, that's perfect. However, if we replace it with
`VtkHdfV2PostProcessor`, it will be problematic because these two services do
not use the same options! The `fileset-size` and `binary-file` options do not
exist in `VtkHdfV2PostProcessor`.

Symbol replacement can also be used in *simple options* that have an array type.

Let's take the following option:

```xml
<!--Fichier AXL-->
<simple name="simple-real-array" type="real[]">
  <description>Tableau de réel</description>
</simple>
```

In the dataset, let's add a symbol:

```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>

  <simple-hydro>
    <simple-real-array>3.0 @DeuxiemeElement@ 3.2 3.3</simple-real-array>
  </simple-hydro>

</case>
```

We can replace the symbol `@DeuxiemeElement@` with `3.1`, or with multiple
values: `3.1 3.11`.


### Assigning a Value to a Symbol in the Command Line {#arcanedoc_execution_commandlineargs_symbol_command}

\note For now, it is necessary to define the environment variable
`ARCANE_REPLACE_SYMBOLS_IN_DATASET`.
```shell
export ARCANE_REPLACE_SYMBOLS_IN_DATASET=1
```

When the dataset contains the desired symbols, we can assign their values during
execution.

This assignment uses the syntax of %Arcane arguments (`-A,`).

Let's assume that we have the symbols `@NamePostProcessor@`,
`@MeshPostProcessor@`, and `@NbTimeInOneFile@` in the dataset.

To assign a value to these symbols, we can run the application like this:

<div class="tabbed">

- <b class="tab-title">Multiple `-A,`</b>
<div>
  ```sh
  ./app \
  -A,NamePostProcessor=Ensight7PostProcessor \
  -A,MeshPostProcessor=Mesh1 \
  -A,NbTimeInOneFile=10 \
  dataset_with_symbols.arc
  ```
</div>

- <b class="tab-title">Unique `-A,`</b>
<div>
  ```sh
  ./app \
  -A,NamePostProcessor=Ensight7PostProcessor,MeshPostProcessor=Mesh1,NbTimeInOneFile=10 \
  dataset_with_symbols.arc
  ```
</div>

</div>


It is also possible to enclose the values in quotes `""`.
This is particularly useful for array types:

```sh
./app \
-A,DeuxiemeElement="3.1 3.11" \
dataset_with_symbols.arc
```

When a symbol is present in the dataset but absent from the command line, the
symbol is simply replaced by an empty string.

\warning In %Arcane, a difference is made between an empty option
`<deltat-cp></deltat-cp>` and an option absent from the dataset. In the first
case, the option's value is **empty** (`String("")`) but is present and
therefore is not replaced by the default value. In the second case, the option's
value is **null** (`String()`) and is therefore replaced by the default value.



## Customization by Option Address {#arcanedoc_execution_commandlineargs_addr}

### Unique Options {#arcanedoc_execution_commandlineargs_addr_unique}

Compared to symbol replacement, this method allows you to keep a valid dataset
without mandatory arguments (but is a bit more verbose).

Both methods also act in the same internal locations, so the possibilities are
the same: it is possible to modify the value of a *simple option*, an
*enumerated option*, or an *extended option*, even if these options are in
*complex options* or *service options*.

For *service options*, as before, we can act on the `name` and `mesh-name`
attributes.

Let's take the first example again:
```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>

  <simple-hydro>
    <deltat-cp>3.0</deltat-cp>
  </simple-hydro>

</case>
```

If we want to modify the value of the `deltat-cp` option, we simply run the
application like this:

```sh
./app \
-A,//simple-hydro/deltat-cp=3.1 \
dataset.arc
```

It is necessary to have the option's address (XPath). To find it, you must parse
the XML elements. Here, the `deltat-cp` option is at the address:
`//case/simple-hydro/deltat-cp`.
Then, we remove the `case/` at the beginning (or `cas/` for datasets in French).
Finally, we can construct the argument to add:

`-A,``//simple-hydro/deltat-cp``=3.1`

As with symbol replacement, we can add quotes:

`-A,//simple-hydro/simple-real-array="3.1 3.11 3.12"`

In the case of attributes, they are designated by an `@` (for the `name`
attribute, we must put `@name` in the address).
If we want to modify the `name` attribute of a service, we can do it like this:

`-A,//simple-hydro/post-processor/@name=VtkHdfV2PostProcessor`

### Multiple Options {#arcanedoc_execution_commandlineargs_addr_multi}

But a problem quickly appears: what if we have multiple options?

```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>

  <simple-hydro>
    <deltat-cp>3.0</deltat-cp>
    <deltat-cp>6.0</deltat-cp>
    <deltat-cp>7.0</deltat-cp>
  </simple-hydro>

</case>
```

In XML, in this kind of case, we use indices.
Thus, for the three values of `deltat-cp`, we address them like this:

`//simple-hydro/deltat-cp[1]`<br>
`//simple-hydro/deltat-cp[2]`<br>
`//simple-hydro/deltat-cp[3]`

\warning There is no error regarding the indices. In XML, indices start at 1,
not 0.

We can also write them like this:

`//simple-hydro[1]/deltat-cp[1]`<br>
`//simple-hydro[1]/deltat-cp[2]`<br>
`//simple-hydro[1]/deltat-cp[3]`

These syntaxes are handled by %Arcane. Thus, to modify the second option, we can
write:

```sh
./app \
-A,//simple-hydro/deltat-cp[2]=6.1 \
dataset.arc
```

Or:

```sh
./app \
-A,//simple-hydro[1]/deltat-cp[2]=6.1 \
dataset.arc
```

### Adding Options {#arcanedoc_execution_commandlineargs_addr_add_option}

Another thing you can do is add options.

If you want to add a fourth `deltat-cp` option, you can add the argument:

`-A,//simple-hydro/deltat-cp[4]=9.0`

It is also possible to add options that are not present in the dataset (but are
present in the AXL):

```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>

  <simple-hydro>
  </simple-hydro>

</case>
```

You could add the options via command line arguments:

```sh
./app \
-A,//simple-hydro/deltat-cp[1]=3.0 \
-A,//simple-hydro/deltat-cp[2]=6.0 \
-A,//simple-hydro/deltat-cp[3]=7.0 \
dataset.arc
```

This also works for *service options*. In this case, it is necessary to add at
least the `name` attribute:

```sh
./app \
-A,//simple-hydro/post-processor/@name=VtkHdfV2PostProcessor \
dataset.arc
```

Then, for example, you can modify the *simple options* of this service or the
`mesh-name` attribute:

```sh
./app \
-A,//simple-hydro/post-processor/@name=VtkHdfV2PostProcessor \
-A,//simple-hydro/post-processor/@mesh-name=Mesh1 \
-A,//simple-hydro/post-processor/max-write-size=50 \
dataset.arc
```

\warning It is not yet possible to add *complex options*.

#### What if we start at index 2? {#arcanedoc_execution_commandlineargs_addr_add_option_default}

If, instead of running our application with these arguments:

```sh
./app \
-A,//simple-hydro/deltat-cp[1]=3.0 \
-A,//simple-hydro/deltat-cp[2]=6.0 \
-A,//simple-hydro/deltat-cp[3]=7.0 \
dataset.arc
```

we ran this:

```sh
./app \
-A,//simple-hydro/deltat-cp[2]=6.0 \
-A,//simple-hydro/deltat-cp[3]=7.0 \
dataset.arc
```

In this case, the option `//simple-hydro/deltat-cp[1]`, which is neither present
in the dataset nor present in the command line arguments, will be added with the
default value.

### Special Syntaxes {#arcanedoc_execution_commandlineargs_addr_multi_special_syntax}

#### The ANY Index {#arcanedoc_execution_commandlineargs_addr_multi_special_syntax_index_any}

The ANY index allows you to process multiple options with different indices at
once.
It is represented by empty brackets: `[]`.

If we take the example above and want to modify all `deltat-cp` values, we can
use the address:

`//simple-hydro/deltat-cp[]`

Launch example:

```sh
./app \
-A,//simple-hydro/deltat-cp[]=2.0 \
dataset.arc
```

In this case, all three `simple-hydro/deltat-cp` options would have the value
`2.0`.

#### The ANY Tag {#arcanedoc_execution_commandlineargs_addr_multi_special_syntax_tag_any}

The ANY tag allows you to change the value of an option present in multiple
locations.
It is represented by an empty address part (two `/` with nothing between them)
(so `//module1/option1/option11`, if you want to replace `option1` with ANY, you
do `//module1//option11`).

\note The ANY tag only replaces one part of the address. To replace multiple
parts, you must include multiple. Let's take the example above and replace
`module1` with ANY: `////option11`

\todo Should we keep this syntax or prefer using `*`? (example:
`//*/*/option11`)
Since `*` can be confusing (it might imply that one could do
`//*/option*/option11` (like a regex) when that is not the case), and the empty
address part seems clearer, the latter was chosen.

Taking the example above, if you want to modify all `deltat-cp[2]` options,
regardless of the module in which this option appears, you can use the address:

`///deltat-cp[2]`

Launch example:

```sh
./app \
-A,///deltat-cp[2]=2.0 \
dataset.arc
```

\warning The ANY tag cannot be present at the end of the address (the address
`//module1/option11/` is invalid).


#### Mixing the Two ANYs {#arcanedoc_execution_commandlineargs_addr_multi_special_syntax_mix_any}

Let's assume we have the dataset:
```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>

  <simple-hydro>
    <checkpoint>
      <deltat-cp>3.0</deltat-cp>
      <print-details-before-cp>true</print-details-before-cp>
    </checkpoint>

    <checkpoint>
      <deltat-cp>6.0</deltat-cp>
      <print-details-before-cp>true</print-details-before-cp>
    </checkpoint>

    <final-time>10.2</final-time>

    <post-processor name="Ensight7PostProcessor">
      <fileset-size>10</fileset-size>
      <binary-file>false</binary-file>
    </post-processor>

    <noidea-service name="StillNoIdea">
      <duration>-1</duration>
      <rewrite>false</rewrite>
    </noidea-service>
  </simple-hydro>

  <pas-simple-hydro>
    <checkpoint>
      <deltat-cp>1.0</deltat-cp>
      <print-details-before-cp>true</print-details-before-cp>
    </checkpoint>

    <checkpoint>
      <deltat-cp>3.0</deltat-cp>
      <print-details-before-cp>true</print-details-before-cp>
    </checkpoint>

    <checkpoint>
      <deltat-cp>7.0</deltat-cp>
      <print-details-before-cp>true</print-details-before-cp>
    </checkpoint>

    <final-time>8.9</final-time>

    <post-processor name="Ensight7PostProcessor">
      <fileset-size>10</fileset-size>
      <binary-file>false</binary-file>
    </post-processor>
  </pas-simple-hydro>

</case>
```

We could start by wanting to modify the `deltat-cp` option of the second
checkpoint in the `pas-simple-hydro` module.
To do this, we can use the argument: <br>
`-A,//pas-simple-hydro/checkpoint[2]/delta-cp=4.0`.

Then, we could modify the first `deltat-cp` of both modules: <br>
`-A,///checkpoint[1]/delta-cp=2.0`.

Next, we might not want any writing between checkpoints: <br>
`-A,///checkpoint[]/print-details-before-cp=false`.

Finally, the services in the `simple-hydro` module must act on `Mesh0` and the
`pas-simple-hydro` services on `Mesh1`: <br>
`-A,//simple-hydro//@mesh-name=Mesh0` <br>
`-A,//pas-simple-hydro//@mesh-name=Mesh1`

Finally, we end up with the command:

```sh
./app \
-A,//pas-simple-hydro/checkpoint[2]/delta-cp=4.0 \
-A,///checkpoint[1]/delta-cp=2.0 \
-A,///checkpoint[]/print-details-before-cp=false \
-A,//simple-hydro//@mesh-name=Mesh0 \
-A,//pas-simple-hydro//@mesh-name=Mesh1 \
dataset.arc
```

### Invalid Syntaxes {#arcanedoc_execution_commandlineargs_addr_invalid_syntax}

For this section, some usage constraints have been implemented:

- Arguments starting with `-A,//` are reserved for this purpose.
- Addresses cannot end with a `/`.
  Invalid example:
```sh
./app \
-A,//simple-hydro/deltat-cp/=3.1 \
dataset.arc
```
- Indices must be integers greater than or equal to 1.
- When an index is present (including the ANY index), a tag must also be
  present.
  Invalid example:
```sh
./app \
-A,//simple-hydro/[2]/option=3.1 \
dataset.arc
```



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_execution_traces
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_general_codingrules
</span> -->
</div>
