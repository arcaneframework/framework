# Dataset (.ARC) {#arcanedoc_core_types_casefile}

[TOC]

## Introduction {#arcanedoc_core_types_casefile_intro}

This chapter describes the structure of a dataset. The dataset is an
[XML](https://www.w3.org/TR/xml) file that contains the values used to configure
the execution of a calculation code.

Here is an example of a dataset in English:

```xml
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane>
    <title>Sod Shock Tube</title>
    <description>This dataset allows testing the simplified Hydro module of
      Arcane
    </description>
    <timeloop>ArcaneHydroLoop</timeloop>
  </arcane>

  <!-- List of meshes (always in English) -->
  <meshes>
    <mesh>
      <filename>tube5x5x100.ice</filename>
    </mesh>
  </meshes>

  <functions>
    <table name="table-dt" parameter="time" value="real" interpolation="linear">
      <value>
        <x>0.0</x>
        <y>1.0e-3</y>
      </value>
      <value>
        <x>1.0e-2</x>
        <y>1.0e-5</y>
      </value>
    </table>
  </functions>

  <simple-hydro>
    <deltat-init>0.001</deltat-init>
    <deltat-min>0.0001</deltat-min>
    <deltat-max>0.01</deltat-max>
    <final-time>0.2</final-time>
  </simple-hydro>

</case>
```

A dataset consists of four parts:

- the `<arcane>` element, which allows, among other things, configuring the time
  loop and the active modules (see
  \ref arcanedoc_core_types_casefile_arcaneelement)
- the mesh element (`<meshes>`), which allows describing the mesh (see
  \ref arcanedoc_core_types_casefile_meshes).
- the element containing functions (`<functions>` or `<fonctions>`) (see
  \ref arcanedoc_core_types_casefile_functions)
- the remaining elements concern the options of the different code modules.

Only the `<arcane>` tag is required. The other tags are optional.

\warning Whitespace characters in attributes are forbidden in the XML standard.
Therefore, the dataset is invalid if they are present. However, parsers may
tolerate them. Whitespace characters at the beginning and end of tags are
significant and can change the meaning or render a dataset invalid. For example,
`<file> Toto</file>` indicates that the mesh file contains a space before the
characters 'Toto'.

\note For historical reasons, it is tolerated to have spaces at the beginning or
end of a tag's text in the case of simple dataset options (see 
\ref arcanedoc_core_types_axl_caseoptions_options_simple). In this case, these
spaces are ignored. For example, `<deltat>  25.0  </deltat>` is valid. These
spaces are only allowed in module and service options, not in tags specific to
%Arcane (such as `<arcane>`, `<functions>`, `<meshes>`, ...).

## <arcane> Element {#arcanedoc_core_types_casefile_arcaneelement}

This element contains information about the time loop used and the list of
active modules. The content of this element is the first thing read in the
dataset. The following elements are possible:

```xml
<arcane>
  <title>Sod Shock Tube</title>
  <description>This dataset allows testing the simplified Hydro module of
    Arcane
  </description>
  <timeloop>ArcaneHydroLoop</timeloop>
  <modules>
    <module name="Hydro" actif="true"/>
    <module name="PostProcessing" actif="false"/>
  </modules>
  <configuration>
    <parameter name="NotParallel" value="false"/>
    <parameter name="NotCheckpoint" value="true"/>
  </configuration>
</arcane>
```

The following table lists the possible elements:

<table>
<tr>
<th>English element</th>
<th>French element</th>
<th>Description</th>
</tr>
<tr>

<td><b>timeloop</b></td>
<td><b>boucle-en-temps</b></td>
<td> Name of the time loop used. This name must correspond to a time loop
available in the code's configuration file.
</td>
</tr>

<tr>
<td><b>title</b></td>
<td><b>titre</b></td>
<td> Title of the dataset. Purely informational.
</td>
</tr>

<tr>
<td><b>description</b></td>
<td><b>description</b></td>
<td> Description of the test case. Purely informational.
</td>
</tr>

<tr>
<td><b>modules</b></td>
<td><b>modules</b></td>
<td> List of modules with their activation status. This tag is used to indicate
whether an optional module is active or not. By default, optional modules are
not active. It is a list of `<module>` elements as follows:

```xml
<module name="Module1" active='true'/>
<module name="Module2" active='false'/>
```
</td>
</tr>

<tr>
<td><b>services</b></td>
<td><b>services</b></td>
<td> List of singleton services with their activation status (which defaults to
*true*). It is a list of `<service>` elements as follows:

```xml
<service name="Service1" active='true'/>
<service name="Service2" active='false'/>
<service name="Service3"/>
```
In the previous example, the services named 'Service1' and 'Service3' will be
loaded.
</td>
</tr>

<tr>
<td><b>configuration</b></td>
<td><b>configuration</b></td>
<td> List of configuration parameters. These parameters are not read by %Arcane
but can be used, for example, by the code's launch procedure. Each parameter is
in the following format:

```xml
<parameter name="Param1" value='value1'/> <!-- English -->
<parametre name="Param1" value='value1'/> <!-- French -->
```
</td>
</tr>
</table>

## Meshes (tag <meshes>) {#arcanedoc_core_types_casefile_meshes}

Meshes are managed by the `ArcaneCaseMeshService` service. Possible values are
described in the page \ref axldoc_service_ArcaneCaseMeshService_arcane_impl. It
is possible to specify multiple meshes. For example:

~~~xml
<meshes>
  <mesh>
    <filename>sod.vtk</filename>
  </mesh>
  <mesh>
    <filename>plancher.msh</filename>
  </mesh>
</meshes>
~~~

There is another possibility to specify meshes. This possibility is declared
obsolete and should only be used by existing codes. For these codes, the
`<mesh>` tag (or `<maillage>` in French) is used to specify mesh information.
For example:

~~~xml
<maillage>
  <fichier>sod.vtk</fichier>
</maillage>
<maillage>
  <fichier>plancher.msh</fichier>
</maillage>
~~~

## Functions (tags <fonctions> or <functions>) {#arcanedoc_core_types_casefile_functions}

It is possible to define functions in the dataset that are used to vary the
values of an option based on physical time or iteration. The set of functions is
defined in the `<fonctions>` tag if the language is French or `<functions>` if
the language is English.

\note In the rest of the document, only English terms will be used to improve
readability.

A function must have a unique name that is used by the option to reference it.
The following example shows how to define a `table-dt` function and use it as a
reference in the `<my-option>` option:

~~~{xml}
<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
 </arcane>

 <functions>
  <table name="table-dt" parameter="time" value="real" interpolation="linear">
   ...
  </table>
 </functions>
 <my-module>
  <my-option function="table-dt">0.01</function>
 </my-module>
~~~

Functions take a single argument as a parameter and can return the type expected
by the option that uses them. The two possible values for the parameters are:

- physical time. In this case, the parameter type is a `real`.
- iteration number. In this case, the parameter type is an integer.

At the beginning of each iteration, %Arcane automatically updates the dataset
options that reference a function (via the method
\arcane{ICaseMng::updateOptions()}).

The `<functions>` tag allows defining two types of functions:

- lookup tables specified directly in the dataset. These are continuous or
  piecewise linear functions.
- external functions written in C#. In this case, any type of function whose
  signature matches the type expected by the option can be defined. Chapter
  \ref arcanedoc_wrapping_csharp_casefunction indicates how to define and use
  these functions.

### Lookup Table Syntax

A lookup table is a continuous or piecewise linear function defined by a set of
`(X,Y)` pairs. For example:

~~~{xml}
 <!-- Example in English -->
 <functions>
  <table name="table-dt" parameter="time" value="real" interpolation="linear">
   <value><x>0.0</x><y>1.0e-3</y></value>
   <value><x>1.0e-2</x><y>1.0e-5</y></value>
  </table>
 </functions>
~~~

~~~{xml}
 <!-- Example in French -->
 <table nom='test-time-real-2' parametre='temps' valeur='reel' interpolation='lineaire'>
  <valeur> <x>0.0</x> <y>3.0</y> </valeur>
  <valeur> <x>4.0</x> <y>9.0</y> </valeur>
  <valeur> <x>5.0</x> <y>7.</y> </valeur>
  <valeur> <x>6.0</x> <y>2.0</y> </valeur>
  <valeur> <x>10.0</x><y>-1.0</y> </valeur>
  <valeur> <x>14.0</x><y>-3.0</y> </valeur>
 </table>
~~~

The `(X,Y)` pairs must be sorted by increasing value of `X`. If the parameter
value is smaller or larger than the first or last value of the lookup table, the
last value is taken. In the previous example for the lookup table
`test-time-real-2`, if `X<0.0` then `3.0` is returned, and if `X>14.0` then
`-3.0` is returned.

Lookup tables have the following attributes:

<table>
<tr>
<th>English Name</th>
<th>French Name</th>
<th>Type</th>
<th>Description</th>
</tr>

<tr>
<td>nom</td>
<td>name</td>
<td>string</td>
<td>Name of the lookup table.
</td>

</tr>

<tr>
<td>parameter</td>
<td>parametre</td>
<td>string</td>
<td>Parameter type. Possible values are `time` (`temps` in French) for a
parameter that is physical time or `iteration` for a parameter that is the
current iteration number.
</td>
</tr>

<tr>
<td>value</td>
<td>valeur</td>
<td>string</td>
<td>Type of the lookup table return value. Possible values are `real`,
`integer`, `real3`, `string`, or `bool` (respectively `reel`, `entier`,
`reel3`, `string`, and `bool` in French).
</td>
</tr>

<tr>
<td>interpolation</td>
<td>interpolation</td>
<td>string</td>
<td>Possible values are `linear` or `constant` (`lineaire` or
`constant-par-morceaux` in French). If the interpolation is constant, the
returned value is the corresponding `Y` value that matches the `X` immediately
below the parameter value. If the interpolation is linear, a linear
interpolation is performed between `(X1,Y1)` and `(X2,Y2)`, where `X1` is the
value of `X` immediately below the parameter and `X2` is the next value in the
lookup table. For the previous lookup table (`test-time-real-2`) example, if
`X` equals `4.5`, then `9.0` is returned if the interpolation is piecewise
constant, and `8.0` (i.e., `Y1 + (X-X1)*(Y2-Y1)/(X2-X1)` <=> `9.0 + (4.5-4.0)*(7.0-9.0)/(5.0-4.0)`)
if the interpolation is linear.
</td>
</tr>

<tr>
<td>comul</td>
<td>comul</td>
<td>string</td>
<td>Multiplier coefficient for the value. This attribute is optional and must be
of the same type as the function's value. If present, the function's value is
multiplied by the value of this attribute (for a `Real3`, the multiplication
is done component by component).
</td>
</tr>

<tr>
<td>deltat-coef</td>
<td>deltat-coef</td>
<td>real</td>
<td>Multiplier coefficient for physical time. This attribute is optional and is
used to multiply the current time step value (see
\arcane{ICaseMng::updateOptions()}).
</td>
</tr>

</table>

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_caseoptions_default_values
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_codeconfig
</span>
</div>
