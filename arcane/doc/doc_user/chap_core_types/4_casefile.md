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

There are two ways to specify the meshes of a computation in a dataset:

- the modern method, based on mesh services, which uses the `<meshes>`
  element. This is the recommended method and new users should use it (see
  \ref arcanedoc_core_types_casefile_meshes_services).
- the legacy method, which uses one or several `<mesh>` elements (or
  `<maillage>` in French). This method is **obsolete** and should only be
  used by existing codes (see
  \ref arcanedoc_core_types_casefile_meshes_legacy).

\note If both methods are used in the same dataset, the `<meshes>` element
takes precedence: the meshes are then created by the mesh services and the
legacy `<mesh>` elements are ignored.

### Specifying Meshes with Services (tag <meshes>) {#arcanedoc_core_types_casefile_meshes_services}

With this method, the meshes are managed by services. The `<meshes>` element
is read by the
\ref axldoc_service_ArcaneCaseMeshMasterService_arcane_impl "ArcaneCaseMeshMasterService"
service. This element is always
in English, even for a French dataset (i.e. when `xml:lang="fr"`), because it
is read before the other options of the dataset: the meshes have to be
created before the modules and the services are initialized, as the latter
may require them.

The `<meshes>` element contains one or several `<mesh>` elements. Each
`<mesh>` element is an instance of a service implementing the
`Arcane::ICaseMeshService` interface. By default, the
\ref axldoc_service_ArcaneCaseMeshService_arcane_impl "ArcaneCaseMeshService"
service is used and its options are described in the
following sections (the complete and up-to-date list is also available in the
page \ref axldoc_service_ArcaneCaseMeshService_arcane_impl).

The meshes are named `Mesh0`, `Mesh1`, ... in the order of appearance of the
`<mesh>` elements. The first mesh (`Mesh0`) is the default mesh of the
sub-domain and must always be created. The other meshes may be null on some
or all sub-domains (for example when a `mesh-builder-customiser` service is
used).

\note When several meshes are defined, a module or a service can be
associated with a given mesh using the `mesh-name` attribute of the
corresponding element of the dataset (see
\ref arcanedoc_core_types_axl_caseoptions_options_service).

The following example specifies two meshes, the first one read from a file
and the second one created by a generator:

~~~xml
<meshes>
  <mesh>
    <filename>sod.vtk</filename>
  </mesh>
  <mesh>
    <generator name="Cartesian3D">
      <origin>1.0 2.0 3.0</origin>
      <x><n>100</n><length>2.0</length></x>
      <y><n>15</n><length>2.0</length></y>
      <z><n>10</n><length>4.0</length></z>
    </generator>
  </mesh>
</meshes>
~~~

#### Options of the <mesh> Element {#arcanedoc_core_types_casefile_meshes_meshoptions}

The following table lists the possible elements of `<mesh>` (options of the
default \ref axldoc_service_ArcaneCaseMeshService_arcane_impl "ArcaneCaseMeshService"
service):

<table>
<tr>
<th>Element</th>
<th>Type</th>
<th>Default value</th>
<th>Description</th>
</tr>

<tr>
<td><b>generator</b></td>
<td>service-instance</td>
<td>none</td>
<td> Mesh creation service (an implementation of the
`Arcane::IMeshBuilder` interface). One and only one of the `<generator>` or
`<filename>` elements must be specified. The available generators are
described in \ref arcanedoc_core_types_casefile_meshes_generators.
</td>
</tr>

<tr>
<td><b>filename</b></td>
<td>string</td>
<td>none</td>
<td> Name of the mesh file to read. The reader is automatically detected
from the file extension (see
\ref arcanedoc_core_types_casefile_meshes_readers). One and only one of the
`<generator>` or `<filename>` elements must be specified.
</td>
</tr>

<tr>
<td><b>partitioner</b></td>
<td>string</td>
<td>`DefaultPartitioner`</td>
<td> Name of the partitioning service to use for the initial decomposition.
If the value is `External`, it is considered that the partitioning has
already been performed before the code is launched. In this case, the
already partitioned mesh files must be available and be named
`CPU%05d.&lt;ext&gt;` where `%05d` is the rank of the process (zero-padded
on 5 digits) and `&lt;ext&gt;` the extension of the mesh file.
</td>
</tr>

<tr>
<td><b>nb-ghost-layer</b></td>
<td>integer</td>
<td>1</td>
<td> Number of ghost cell layers to add to the mesh when it is partitioned.
</td>
</tr>

<tr>
<td><b>ghost-layer-builder-version</b></td>
<td>integer</td>
<td>3</td>
<td> Version of the ghost cell manager to use. This option is internal to
%Arcane.
</td>
</tr>

<tr>
<td><b>face-numbering-version</b></td>
<td>integer</td>
<td>none</td>
<td> Version of the calculation of the unique identifiers of the faces.
This option is internal to %Arcane.
</td>
</tr>

<tr>
<td><b>edge-numbering-version</b></td>
<td>integer</td>
<td>none</td>
<td> Version of the calculation of the unique identifiers of the edges.
This option is only used for 3D meshes. This option is internal to %Arcane.
</td>
</tr>

<tr>
<td><b>cell-dimension-kind</b></td>
<td>enumeration</td>
<td>`mono-dimension`</td>
<td> Indicates whether the mesh is 'non-manifold' or multi-dimensional. A
mesh of this type contains cells of different dimensions. Possible values
are `mono-dimension`, `multi-dimension` and `non-manifold`.

\warning This option is experimental. Do not use outside of %Arcane.
</td>
</tr>

<tr>
<td><b>specific-reader</b></td>
<td>service-instance</td>
<td>none</td>
<td> Specific mesh reading service (an implementation of the
`Arcane::ICaseMeshReader` interface). This service can be specified to avoid
the automatic detection of the reader or to add specific options to a given
reader.
</td>
</tr>

<tr>
<td><b>mesh-builder-customiser</b></td>
<td>service-instance</td>
<td>none</td>
<td> Service (an implementation of the `Arcane::IMeshBuilderCustomiser`
interface) used to customise the creation of the mesh. It may be used for
example to create the mesh on a subset of the ranks (e.g. the
\ref axldoc_service_ArcaneParallelMngSplitterMeshBuilderCustomiser_arcane_impl "ArcaneParallelMngSplitterMeshBuilderCustomiser"
service with its `subset-ratio` option).
</td>
</tr>

<tr>
<td><b>subdivider</b></td>
<td>service-instance</td>
<td>\ref axldoc_service_ArcaneBasicMeshSubdividerService_arcane_impl "ArcaneBasicMeshSubdivider"</td>
<td> Service (an implementation of the `Arcane::IMeshSubdivider` interface)
allowing the subdivision of cells. The default service has one option,
`nb-subdivision`, which is the number of subdivisions to apply to each cell.
</td>
</tr>

<tr>
<td><b>initialization</b></td>
<td>complex</td>
<td>none</td>
<td> Initialization information for the mesh variables. It contains a list
of `<variable>` elements (see below).
</td>
</tr>

</table>

The `<initialization>` element contains zero or several `<variable>` elements
with the following options:

<table>
<tr>
<th>Element</th>
<th>Type</th>
<th>Description</th>
</tr>

<tr>
<td><b>name</b></td>
<td>string</td>
<td> Name of the variable to initialize.</td>
</tr>

<tr>
<td><b>value</b></td>
<td>string</td>
<td> Value of the variable. The string must be convertible to the type of
the variable.</td>
</tr>

<tr>
<td><b>group</b></td>
<td>string</td>
<td> Name of the group of items on which the values are set.</td>
</tr>

</table>

For example:

~~~xml
<meshes>
  <mesh>
    <generator name="Cartesian3D"/>
    <initialization>
      <variable><name>Density</name><value>1.0</value><group>ZG</group></variable>
      <variable><name>Density</name><value>0.125</value><group>ZD</group></variable>
    </initialization>
  </mesh>
</meshes>
~~~

#### Mesh Generators {#arcanedoc_core_types_casefile_meshes_generators}

The `<generator>` element specifies an implementation of the
`Arcane::IMeshBuilder` interface. The name of the service is given by the
`name` attribute. The following generators are provided by %Arcane (the list
of available generators depends on the compile options):

**\ref axldoc_service_Cartesian2DMeshGenerator_arcane_std "Cartesian2D"** — 2D Cartesian mesh generator.

If the division in X and Y is not specified, the number of subdomains must
be a perfect square (1, 4, 9, 16, 25, ...). If only the division in X or Y
is specified, this specified value must be a divisor of the number of
subdomains.

<table>
<tr>
<th>Element</th>
<th>Type</th>
<th>Default value</th>
<th>Description</th>
</tr>

<tr>
<td><b>origin</b></td>
<td>real2</td>
<td>none</td>
<td> Mesh origin.</td>
</tr>

<tr>
<td><b>nb-part-x</b></td>
<td>integer</td>
<td>0</td>
<td> Number of subdomains in X. If absent or if the value `0` is specified,
the X division is calculated dynamically.</td>
</tr>

<tr>
<td><b>nb-part-y</b></td>
<td>integer</td>
<td>0</td>
<td> Number of subdomains in Y. If absent or if the value `0` is specified,
the Y division is calculated dynamically.</td>
</tr>

<tr>
<td><b>generate-sod-groups</b></td>
<td>bool</td>
<td>false</td>
<td> If *true*, groups are generated for a Sod test case (internal to
%Arcane).</td>
</tr>

<tr>
<td><b>face-numbering-version</b></td>
<td>integer</td>
<td>4</td>
<td> Version of the algorithm for face numbering. Version 1 corresponds to
unstructured numbering. Version 4 corresponds to Cartesian numbering.</td>
</tr>

<tr>
<td><b>x</b></td>
<td>complex</td>
<td>none</td>
<td> Description of the mesh blocks in X (at least one). See below.</td>
</tr>

<tr>
<td><b>y</b></td>
<td>complex</td>
<td>none</td>
<td> Description of the mesh blocks in Y (at least one). See below.</td>
</tr>

</table>

**\ref axldoc_service_Cartesian3DMeshGenerator_arcane_std "Cartesian3D"** — 3D Cartesian mesh generator.

It has the same options as
\ref axldoc_service_Cartesian2DMeshGenerator_arcane_std "Cartesian2D"
(with a `real3` origin), plus the
following ones:

<table>
<tr>
<th>Element</th>
<th>Type</th>
<th>Default value</th>
<th>Description</th>
</tr>

<tr>
<td><b>nb-part-x</b> / <b>nb-part-y</b> / <b>nb-part-z</b></td>
<td>integer</td>
<td>none</td>
<td> Number of sub-domains in X, Y and Z.</td>
</tr>

<tr>
<td><b>edge-numbering-version</b></td>
<td>integer</td>
<td>2</td>
<td> Version of the algorithm for edge numbering. Only value 2 is currently
valid. In this case, if edges are enabled, the maximum number of nodes is
limited to 2^31.</td>
</tr>

<tr>
<td><b>z</b></td>
<td>complex</td>
<td>none</td>
<td> Description of the mesh blocks in Z (at least one). See below.</td>
</tr>

</table>

For \ref axldoc_service_Cartesian2DMeshGenerator_arcane_std "Cartesian2D" and
\ref axldoc_service_Cartesian3DMeshGenerator_arcane_std "Cartesian3D", the
mesh is described by blocks in each
direction. Each `<x>`, `<y>` or `<z>` element describes one block and can be
repeated to describe several consecutive blocks:

~~~xml
<x>
  <n>10</n>
  <length>2.0</length>
  <progression>1.0</progression>
</x>
~~~

<table>
<tr>
<th>Element</th>
<th>Type</th>
<th>Default value</th>
<th>Description</th>
</tr>

<tr>
<td><b>n</b></td>
<td>integer</td>
<td>none</td>
<td> Number of cells of the block.</td>
</tr>

<tr>
<td><b>length</b></td>
<td>real</td>
<td>none</td>
<td> Length of the block.</td>
</tr>

<tr>
<td><b>progression</b></td>
<td>real</td>
<td>1.0</td>
<td> Geometric progression of the cell sizes inside the block.</td>
</tr>

</table>

**\ref axldoc_service_Sod3DMeshGenerator_arcane_std "Sod3D"** — Sod shock tube mesh generator (3D).

<table>
<tr>
<th>Element</th>
<th>Type</th>
<th>Description</th>
</tr>

<tr>
<td><b>x</b></td>
<td>integer</td>
<td> Number of cells in X.</td>
</tr>

<tr>
<td><b>y</b></td>
<td>integer</td>
<td> Number of cells in Y.</td>
</tr>

<tr>
<td><b>z</b></td>
<td>integer</td>
<td> Number of cells in Z.</td>
</tr>

</table>

**\ref axldoc_service_HoneyComb2DMeshGenerator_arcane_std "HoneyComb2D"** — 2D hexagonal mesh generator.

<table>
<tr>
<th>Element</th>
<th>Type</th>
<th>Description</th>
</tr>

<tr>
<td><b>origin</b></td>
<td>real2</td>
<td> Mesh origin.</td>
</tr>

<tr>
<td><b>pitch-size</b></td>
<td>real</td>
<td> Pitch value, which is the distance between two opposite sides of a
hexagon. Must be strictly positive.</td>
</tr>

<tr>
<td><b>nb-layer</b></td>
<td>integer</td>
<td> Number of layers of hexagons. If 1, there is only the central hexagon.
Must be strictly positive.</td>
</tr>

</table>

**\ref axldoc_service_HoneyComb3DMeshGenerator_arcane_std "HoneyComb3D"** — 3D hexagonal mesh generator.

It has the same options as
\ref axldoc_service_HoneyComb2DMeshGenerator_arcane_std "HoneyComb2D"
(the origin is 2D: the height values
are provided by the `heights` option), plus the following one:

<table>
<tr>
<th>Element</th>
<th>Type</th>
<th>Description</th>
</tr>

<tr>
<td><b>heights</b></td>
<td>real[]</td>
<td> List of cell heights. This array must have at least 2 values. The
values must be increasing. The number of cell layers in height is equal to
the number of elements in this array minus 1.</td>
</tr>

</table>

Here is a complete example using a generator:

~~~xml
<meshes>
  <mesh>
    <generator name="Cartesian2D" >
      <nb-part-x>2</nb-part-x>
      <nb-part-y>2</nb-part-y>
      <origin>0.0 0.0</origin>
      <x><n>5</n><length>5.0</length></x>
      <y><n>3</n><length>3.0</length></y>
    </generator>
  </mesh>
</meshes>
~~~

#### Mesh Files and Readers {#arcanedoc_core_types_casefile_meshes_readers}

When the mesh is read from a file (element `<filename>`), the mesh reading
service is automatically detected from the extension of the file. The
following readers are provided by %Arcane (the list of available readers
depends on the compile options):

<table>
<tr>
<th>Extension(s)</th>
<th>Service</th>
<th>Description</th>
</tr>

<tr>
<td>`vtk`</td>
<td>`VtkLegacyCaseMeshReader`</td>
<td> Legacy (structured) VTK format.</td>
</tr>

<tr>
<td>`vtk` / `vtu`</td>
<td>\ref axldoc_service_VtkPolyhedralMeshIO_arcane_std "VtkPolyhedralCaseMeshReader"</td>
<td> VTK/VTU format with support for polyhedral cells. Options:
`print-mesh-infos` (bool, default *false*) and `print-debug-infos` (bool,
default *false*).</td>
</tr>

<tr>
<td>`vtu`</td>
<td>`VtuCaseMeshReader`</td>
<td> XML VTK (VTU) format.</td>
</tr>

<tr>
<td>`msh`</td>
<td>`MshCaseMeshReader`</td>
<td> Gmsh format.</td>
</tr>

<tr>
<td>`med`</td>
<td>`MEDCaseMeshReader`</td>
<td> MED format (requires HDF5 support).</td>
</tr>

<tr>
<td>`unf` / `mli` / `mli2` / `ice` / `uns` / `unv`</td>
<td>`LimaCaseMeshReader`</td>
<td> Lima formats (requires Lima support).</td>
</tr>

</table>

The automatic detection can be avoided and the reader configured with the
`<specific-reader>` element. For example:

~~~xml
<meshes>
  <mesh>
    <filename>faultx4_3x3x2.vtk</filename>
    <specific-reader name="VtkPolyhedralCaseMeshReader">
      <print-mesh-infos>true</print-mesh-infos>
    </specific-reader>
  </mesh>
</meshes>
~~~

### Legacy Mesh Specification (tag <mesh> or <maillage>) {#arcanedoc_core_types_casefile_meshes_legacy}

\warning This method is obsolete: it is only kept for compatibility with
existing codes. New codes and new users should use the method based on mesh
services (see \ref arcanedoc_core_types_casefile_meshes_services).

With this method, the mesh information is specified with one or several
`<mesh>` elements (or `<maillage>` if the language of the dataset is French,
i.e. when `xml:lang="fr"`). Contrary to the `<meshes>` element, the names of
the elements and the attributes of this method depend on the language of the
dataset: in the following, both the English and the French names are given.

The first `<mesh>` element describes the default mesh (`Mesh0`). The
additional meshes (i.e. the second, third, ... `<mesh>` elements) are named
`Mesh1`, `Mesh2`, ... or `DualMesh1`, `DualMesh2`, ... if the `dual`
attribute is set to *true*.

The following example specifies a mesh read from a file, with an internal
partitioning:

~~~xml
<mesh>
  <file internal-partition="true">mesh1.med</file>
</mesh>
~~~

#### The <mesh> / <maillage> Element

The `<mesh>` element accepts the following attributes:

<table>
<tr>
<th>English attribute</th>
<th>French attribute</th>
<th>Type</th>
<th>Default value</th>
<th>Description</th>
</tr>

<tr>
<td>`amr`</td>
<td>`amr`</td>
<td>bool</td>
<td>false</td>
<td> If *true*, the mesh supports adaptive mesh refinement (AMR).</td>
</tr>

<tr>
<td>`amr-type`</td>
<td>`amr-type`</td>
<td>integer</td>
<td>0</td>
<td> Type of AMR: `0` none, `1` cell-based refinement (`Cell`), `2`
patch-based refinement (`Patch`), `3` patch on Cartesian mesh only
(`PatchCartesianMeshOnly`). If `amr` is *true* and this attribute is `0`, the
`Cell` type is used.</td>
</tr>

<tr>
<td>`dimension`</td>
<td>`dimension`</td>
<td>integer</td>
<td>none</td>
<td> Forces the dimension of the mesh (`1`, `2` or `3`). Useful when the
mesh format can not determine it.</td>
</tr>

<tr>
<td>`nb-ghostlayer`</td>
<td>`nb-ghostlayer`</td>
<td>integer</td>
<td>none</td>
<td> Number of ghost cell layers to add to the mesh when it is
partitioned.</td>
</tr>

<tr>
<td>`ghostlayer-builder-version`</td>
<td>`ghostlayer-builder-version`</td>
<td>integer</td>
<td>none</td>
<td> Version of the ghost layer builder.</td>
</tr>

<tr>
<td>`face-numbering-version`</td>
<td>`face-numbering-version`</td>
<td>integer</td>
<td>none</td>
<td> Version of the algorithm used to compute the unique identifiers of the
faces.</td>
</tr>

<tr>
<td>`dual`</td>
<td>`dual`</td>
<td>bool</td>
<td>false</td>
<td> If *true*, the additional meshes are named `DualMeshN` instead of
`MeshN`.</td>
</tr>

<tr>
<td>`use-unit`</td>
<td>`utilise-unite`</td>
<td>bool</td>
<td>true</td>
<td> Only for the Lima format. If *true*, the length units of the Lima
files are taken into account.</td>
</tr>

</table>

#### The <file> / <fichier> Element

The `<file>` element (or `<fichier>` in French) specifies the mesh file to
read. The value of the element is the name of the mesh file. It accepts the
following attributes:

<table>
<tr>
<th>English attribute</th>
<th>French attribute</th>
<th>Type</th>
<th>Default value</th>
<th>Description</th>
</tr>

<tr>
<td>`format`</td>
<td>`format`</td>
<td>string</td>
<td>none</td>
<td> Forces the format of the mesh file (i.e. its extension, e.g. `mli`).
By default, the format is deduced from the extension of the file name.</td>
</tr>

<tr>
<td>`internal-partition`</td>
<td>`internal-partition`</td>
<td>bool</td>
<td>false</td>
<td> If *true*, the mesh is partitioned internally by %Arcane: only one
process reads the mesh file and the mesh is then distributed among the
processes. If *false* or absent, each process reads its own part from
pre-partitioned files named `CPU%05d.&lt;ext&gt;`.</td>
</tr>

<tr>
<td>`partitioner`</td>
<td>`partitioner`</td>
<td>string</td>
<td>none</td>
<td> Name of the partitioning service to use when `internal-partition` is
*true*.</td>
</tr>

<tr>
<td>`unique`</td>
<td>`unique`</td>
<td>bool</td>
<td>false</td>
<td> If *true*, a unique mesh file is used even in parallel (no
`CPU%05d.&lt;ext&gt;` files).</td>
</tr>

<tr>
<td>`cut_N`</td>
<td>`cut_N`</td>
<td>string</td>
<td>none</td>
<td> Directory containing the pre-partitioned mesh files for `N`
sub-domains, where `N` is the number of sub-domains (e.g. `cut_4` for 4
sub-domains).</td>
</tr>

</table>

#### The <meshgenerator> Element

The `<meshgenerator>` element (the name is the same in English and in French)
allows generating a mesh without a mesh file. It contains one and only one of
the following elements:

**`<cartesian>`** — Cartesian mesh generator. The dimension of the mesh is
deduced from the presence of the `<lz>` elements (2D if absent, 3D if
present). It accepts the following elements:

<table>
<tr>
<th>Element</th>
<th>Type</th>
<th>Description</th>
</tr>

<tr>
<td><b>origine</b></td>
<td>real3</td>
<td> Origin of the mesh (only the two first components are used for a 2D
mesh).</td>
</tr>

<tr>
<td><b>nsd</b></td>
<td>integer[]</td>
<td> Number of sub-domains in each direction. The number of values must be
equal to the dimension of the mesh.</td>
</tr>

<tr>
<td><b>lx</b></td>
<td>real</td>
<td> Length of a block in X. This element can be repeated to describe
several consecutive blocks. It has a mandatory attribute `nx` (number of
cells of the block) and an optional attribute `prx` (geometric progression of
the cell sizes, default 1.0).</td>
</tr>

<tr>
<td><b>ly</b></td>
<td>real</td>
<td> Same as `<lx>` for the Y direction (attributes `ny` and `pry`).</td>
</tr>

<tr>
<td><b>lz</b></td>
<td>real</td>
<td> Same as `<lx>` for the Z direction (attributes `nz` and `prz`). Only
for 3D meshes.</td>
</tr>

<tr>
<td><b>face-numbering-version</b></td>
<td>integer</td>
<td> Version of the algorithm used to compute the unique identifiers of the
faces.</td>
</tr>

<tr>
<td><b>edge-numbering-version</b></td>
<td>integer</td>
<td> Version of the algorithm used to compute the unique identifiers of the
edges.</td>
</tr>

</table>

**`<sod>`** — Sod shock tube mesh generator. The dimension of the mesh is
deduced from the elements present: 1D if only `<x>` is specified, 2D if
`<x>` and `<y>` are specified, 3D if `<x>`, `<y>` and `<z>` are specified.
It accepts the attribute `zyx` (bool, default *false*) to generate the mesh
in the ZYX order, and the following elements:

<table>
<tr>
<th>Element</th>
<th>Type</th>
<th>Description</th>
</tr>

<tr>
<td><b>x</b></td>
<td>integer</td>
<td> Number of cells in X. Optional attribute `delta` (real): space step in
X.</td>
</tr>

<tr>
<td><b>y</b></td>
<td>integer</td>
<td> Number of cells in Y. Optional attribute `delta` (real): space step in
Y.</td>
</tr>

<tr>
<td><b>z</b></td>
<td>integer</td>
<td> Number of cells in Z. Optional attribute `delta` (real): space step in
Z. Optional attribute `total` (bool): if *true*, the value is the total
number of layers for all the processes.</td>
</tr>

<tr>
<td><b>random-coef</b></td>
<td>real</td>
<td> Random perturbation coefficient of the nodes.</td>
</tr>

</table>

The expected syntax is for example
`<meshgenerator><sod><x>nx</x><y>ny</y><z>nz</z></sod></meshgenerator>`.

**`<simple>`** — simple mesh generator, mainly used for tests. It accepts one
element:

<table>
<tr>
<th>Element</th>
<th>Type</th>
<th>Description</th>
</tr>

<tr>
<td><b>mode</b></td>
<td>integer</td>
<td> Type of mesh to generate (value between 1 and 4).</td>
</tr>

</table>

#### The <partitioner> / <partitionneur> Element

The value of this element is the name of the partitioning service to use for
the initial decomposition (e.g.
\ref axldoc_service_MetisMeshPartitioner_arcane_std "Metis",
\ref axldoc_service_PTScotchMeshPartitioner_arcane_std "PTScotch",
\ref axldoc_service_ZoltanMeshPartitioner_arcane_std "Zoltan"). If this
element is absent, the default partitioner of %Arcane is used. It accepts the
following attribute:

<table>
<tr>
<th>English attribute</th>
<th>French attribute</th>
<th>Type</th>
<th>Default value</th>
<th>Description</th>
</tr>

<tr>
<td>`need-basic-partition-first`</td>
<td>`need-basic-partition-first`</td>
<td>bool</td>
<td>false</td>
<td> If *true*, a basic partitioning is done first. This is useful for
parallel partitioners (e.g. ParMetis) when some partitions would otherwise be
empty.</td>
</tr>

</table>

#### The <tied-interfaces> / <interfaces-liees> Element

This element specifies the tied interfaces of the mesh (e.g. welds between
non-conforming parts of the mesh). It contains one or several
`<semi-conform>` elements (or `<semi-conforme>` in French) with the following
attributes:

<table>
<tr>
<th>English attribute</th>
<th>French attribute</th>
<th>Type</th>
<th>Default value</th>
<th>Description</th>
</tr>

<tr>
<td>`slave`</td>
<td>`esclave`</td>
<td>string</td>
<td>none</td>
<td> Name of the face group of the slave interface. This attribute is
mandatory.</td>
</tr>

<tr>
<td>`not-structured`</td>
<td>`non-structure`</td>
<td>bool</td>
<td>false</td>
<td> If *true*, the slave interface is not structured.</td>
</tr>

<tr>
<td>`planar-tolerance`</td>
<td>`planar-tolerance`</td>
<td>real</td>
<td>0.</td>
<td> Tolerance for the planarity of the interface.</td>
</tr>

</table>

\note If no tied interface is specified, the face groups named `SOUDURE` or
`SOUDURES` are automatically used as tied interfaces (if they exist in the
mesh).

#### The <initialisation> Element

This element allows initializing mesh variables with values from the
dataset. It contains a list of `<variable>` elements.

\note For historical reasons, the attributes of the `<variable>` element are
always in French, even for an English dataset.

<table>
<tr>
<th>Attribute</th>
<th>Type</th>
<th>Description</th>
</tr>

<tr>
<td>`nom`</td>
<td>string</td>
<td> Name of the variable to initialize.</td>
</tr>

<tr>
<td>`valeur`</td>
<td>string</td>
<td> Value of the variable. The string must be convertible to the type of
the variable.</td>
</tr>

<tr>
<td>`groupe`</td>
<td>string</td>
<td> Name of the group of items on which the values are set.</td>
</tr>

</table>

#### Legacy Examples

The first example (in English) generates a Sod mesh and initializes mesh
variables on the groups `ZG` and `ZD`:

~~~xml
<mesh>
  <meshgenerator><sod><x>50</x><y>5</y><z>5</z></sod></meshgenerator>
  <initialisation>
    <variable nom="Density" valeur="1." groupe="ZG" />
    <variable nom="Pressure" valeur="1." groupe="ZG" />
    <variable nom="AdiabaticCst" valeur="1.4" groupe="ZG" />
    <variable nom="Density" valeur="0.125" groupe="ZD" />
    <variable nom="Pressure" valeur="0.1" groupe="ZD" />
    <variable nom="AdiabaticCst" valeur="1.4" groupe="ZD" />
  </initialisation>
</mesh>
~~~

The second example (in French) reads a mesh from a file with pre-partitioned
files for 4 sub-domains located in the directory `soudures_3_cut_4`, and
specifies a tied interface:

~~~xml
<maillage utilise-unite="false">
  <fichier cut_4='soudures_3_cut_4'>soudures_3.mli</fichier>
  <interfaces-liees>
    <semi-conforme esclave="INTERFACE1" />
  </interfaces-liees>
</maillage>
~~~

The third example (in French) generates a Cartesian mesh with an explicit
partitioner:

~~~xml
<maillage>
  <meshgenerator>
    <cartesian>
      <origine>0 0 0</origine>
      <nsd>1 1 1</nsd>
      <lx nx="2">2.</lx>
      <ly ny="1">1.</ly>
      <lz nz="1">1.</lz>
    </cartesian>
  </meshgenerator>
  <partitionneur>Titi</partitionneur>
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
