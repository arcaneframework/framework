# Material and Environment Management. {#arcanedoc_materials_manage}

[TOC]

This page describes the management of materials and environments in %Arcane.

The set of classes associated with materials and environments is found
in the Materials namespace of Arcane (see \ref ArcaneMaterials). The simplest
way to use these classes is to use the \a using keyword. For example:

```cpp
#include <arcane/materials/IMeshMaterial.h>

using namespace Arcane::Materials;
```


The set of environments and materials is managed by the class
\arcanemat{IMeshMaterialMng}. An instance of this class is associated with a
mesh \arcane{IMesh}. It is possible to have multiple instances of
\arcanemat{IMeshMaterialMng} per \arcane{IMesh} but this is not used for the
moment.

\warning For now, material and environment management is
incompatible with mesh topology modifications. In particular, this includes
repartitioning due to load balancing.

\note The use of constituents is compatible with mesh topology changes.
Nevertheless, for this support to be active, the method
\arcanemat{IMeshMaterialMng::setMeshModificationNotified(true)} must be called
before creating environments and materials.

An instance of \arcanemat{IMeshMaterialMng} describes a set of environments,
each environment being composed of one or more materials.
The list of environments and materials must also be created during the
calculation initialization and must not evolve afterward. It is also possible to
have the notion of a block above the environment motion, a block comprising one
or more environments. This block notion is optional and is not essential for
using environments or materials.

In the current implementation, environments and materials are associated only
with mesh cells and not with other mesh entities. The list of cells for an
environment or a material is dynamic and can evolve during the calculation.
Similarly, it is possible to have multiple environments and multiple materials
per cell.

\note You must not manually destroy instances of
\arcanemat{IMeshMaterialMng}. They will be automatically destroyed when the
associated mesh is deleted.

The default instance of \arcanemat{IMeshMaterialMng} for a mesh is not
created directly. To retrieve it, you must use the method
\arcanemat{IMeshMaterialMng::getReference()} with the mesh concerned as an
argument. Calling this function triggers the creation of the manager if it has
not already been done. This function has a non-negligible CPU cost, so it is
preferable to store the returned instance rather than calling the function
multiple times.

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialCreate

## Creating materials, environments, and blocks {#arcanedoc_materials_manage_create}

Once the manager is created, you must register the materials and
environments. The first thing to do is to register the material properties. For
now, this is only the material name. We only register the material properties
and not the materials themselves because the latter are created when the
environments are created.

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialCreate2

Once the properties are registered, it is possible to create
the environments, specifying their name and the list of their materials. It
should be noted that two (or more) environments can be composed of the same
material. In this case, %Arcane creates two different instances of the same
material, and these are independent. Thus, in the following example, the
material MAT1 is present in ENV1 and ENV3, which results in two distinct
materials, each with its partial values. Once the environments are created, it
is possible to associate them in blocks. A block is defined by a name, the
associated cell group, and its list of environments, which is therefore fixed.

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialCreate3

Once all environments and blocks are created, you must call
\arcanemat{IMeshMaterialMng::endCreate()}
to signal to the manager that initialization is complete and that it can
allocate the variables. Once this method is called, you must no longer create
blocks, environments, or materials.

It is possible to add environments to a block between its creation
(\arcanemat{IMeshMaterialMng::createBlock()}) and the call to
\arcanemat{IMeshMaterialMng::endCreate()}, via the method
\arcanemat{IMeshMaterialMng::addEnvironmentToBlock()}.

It is possible to remove environments from a block between its creation
(\arcanemat{IMeshMaterialMng::createBlock()}) and the call to
\arcanemat{IMeshMaterialMng::endCreate()}, via the method
\arcanemat{IMeshMaterialMng::removeEnvironmentToBlock()}.

Instances of \arcanemat{IMeshMaterial}, \arcanemat{IMeshEnvironment}, and
\arcanemat{IMeshBlock} remain valid throughout the existence of the
corresponding \arcanemat{IMeshMaterialMng}. They can therefore be kept, and it
is possible to use the equality operator to know if two instances correspond to
the same material.

Information concerning blocks, materials, and environments is saved and can be
reloaded upon restart. Nevertheless, this is not automatic for compatibility
reasons. If you wish to reload the saved information, you must call
\arcanemat{IMeshMaterialMng::recreateFromDump()} and must no longer manually
create the information or call the method
\arcanemat{IMeshMaterialMng::endCreate()}.

Each material \arcanemat{IMeshMaterial}, environment
\arcanemat{IMeshEnvironment}, and block \arcanemat{IMeshBlock} has a
unique identifier, of type \arcane{Int32}, which is accessible via the methods
\arcanemat{IMeshMaterial::id()}, \arcanemat{IMeshEnvironment::id()}, or
\arcanemat{IMeshBlock::id()}. These identifiers start at 0 and are incremented
for each material, environment, or block. For example, with the construction of
the previous example, we have:

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialCreate4

## Adding or removing cells for a material {#arcanedoc_materials_manage_addremovecells}

Once materials and environments are created, it is possible to add or
remove cells for a material. It is not necessary to modify the cells by
environment: Arcane automatically handles recalculating the list of cells for an
environment based on those of its materials.

All modifications are made via the class \arcanemat{MeshMaterialModifier}.
You simply need to create an instance of this class and call the methods
\arcanemat{MeshMaterialModifier::addCells()} or
\arcanemat{MeshMaterialModifier::removeCells()} as many times as necessary. It
should be noted that these two
methods only allow indicating the cells to be added or removed. The effective
modification only takes place when calling
\arcanemat{MeshMaterialModifier::endUpdate()} or when the instance of
\arcanemat{MeshMaterialModifier} is destroyed. Only at that moment are the
partial values updated and accessible.

\note For now, the partial values of materials are automatically
initialized to 0 in the new cells of a material. For performance reasons, this
may no longer be the case, and in that case, the values will not be initialized.

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialAddMat

## Iterating over material cells {#arcanedoc_materials_manage_iteration}

There are three classes to reference the material cell notions:
- \arcanemat{AllEnvCell} is a class that allows access to all
  environments of a cell.
- \arcanemat{EnvCell} corresponds to an environment of a cell and allows access
  to the values of this environment for this cell and to all the values of this
  cell for the materials of this environment.
- \arcanemat{MatCell} corresponds to a value of a material of an environment of
  a cell.

There is a fourth class \arcanemat{ComponentCell} which is not a specific cell
but can represent one of the three types above and allows unifying the
treatments (see \ref arcanedoc_materials_manage_component).

There are two ways to iterate over material cells.

The first is to iterate over all environments, for each environment iterate over
the materials of that environment, and for each of these materials iterate over
their cells. For example:

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialIterEnv

It is also possible to use a block to iterate only over the environments of that
block instead of iterating over all environments:

\snippet MeshMaterialTesterModule_Samples.cc SampleBlockEnvironmentIter

The second way is to traverse all cells of a cell group, and then for each cell
iterate over its environments and over the materials of its environments. To do
this, you can use the macro ENUMERATE_ALLENVCELL, in the same way as the macro
ENUMERATE_CELL, but by specifying the material manager additionally. For
example, if you want to iterate over all cells (group allCells())

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialIterCell

Similarly, by iterating over all cells of a block:

\snippet MeshMaterialTesterModule_Samples.cc SampleBlockMaterialIterCell

There is a third way to iterate over the cells of a material or an environment.
The classes \arcanemat{MatCellVector} and \arcanemat{EnvCellVector} allow
obtaining a list of \arcanemat{MatCell} or \arcanemat{EnvCell} from a cell group
and a material or an environment. The following example shows how to retrieve
the list of cells for the material \a mat and the environment \env corresponding
to the group \a cells to position a variable \a mat_density:

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialIterFromGroup

\note Currently, the classes \arcanemat{MatCellVector} and
\arcanemat{EnvCellVector} are not copyable and are only valid as long as the
material (for \arcanemat{MatCellVector}) or the environment (for
\arcanemat{EnvCellVector}) and the associated cell group do not change.
Furthermore, keeping the information consumes memory and creation has a
calculation time cost proportional to the number of cells in the group.

## Converting Cell to AllEnvCell, MatCell, or EnvCell {#arcanedoc_materials_manage_conversion}

Most methods on the entities return objects of type
\arcane{Cell}. To convert these objects to the type \a \arcanemat{AllEnvCell} in
order to get information about the materials, you must go through an instance of
\arcanemat{CellToAllEnvCellConverter}. This converter can, for example, be
created at the beginning of the function because its creation cost is
negligible.

```cpp
Arcane::Materials::CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);
Arcane::Cell cell = ...;
Arcane::Materials::AllEnvCell allenvcell = all_env_cell_converter[cell];
```

From an \arcanemat{AllEnvCell}, it is possible to directly retrieve a cell for a
given material or environment via
\arcanemat{IMeshMaterial::findMatCell()} or
\arcanemat{IMeshEnvironment::findEnvCell()}. These
methods return a \arcanemat{MatCell} or an \arcanemat{EnvCell} corresponding to
the desired material cell or environment cell. The returned instance may be null
if the material or environment is not present in the cell. For example:

```cpp
Arcane::Materials::AllEnvCell allenvcell = ...;
Arcane::Materials::IMeshMaterial* mat = ...;
Arcane::Materials::MatCell matcell = mat->findMatCell(allenvcell);
if (matcell.null())
  // Material absent from the cell
  ...
Arcane::Materials::IMeshEnvironment* env = ...;
Arcane::Materials::EnvCell envcell = env->findEnvCell(allenvcell);
if (envcell.null())
  // Environment absent from the cell.
  ...
```

## Unification {#arcanedoc_materials_manage_component}

It is possible to treat material and environment cells generically. The class
\arcanemat{ComponentCell} allows this unification and can be used for all types
of cells
\arcanemat{MatCell}, \arcanemat{EnvCell}, or \arcanemat{AllEnvCell}. The macro
ENUMERATE_COMPONENTCELL() allows iterating over cells of this type:

\snippet MeshMaterialTesterModule_Samples.cc SampleComponentIter

The class \arcanemat{ComponentCell} can also be used to index a variable as
would a \arcanemat{MatCell} or an \arcanemat{EnvCell}.

Similarly, the interface \arcanemat{IMeshComponent} is now
the base interface for \arcanemat{IMeshMaterial} and
\arcanemat{IMeshEnvironment}, and the methods
\arcanemat{IMeshMaterialMng::materialsAsComponents()} and
\arcanemat{IMeshMaterialMng::environmentsAsComponents()} allow treating the
lists of materials and environments in the same way as a list of
\arcanemat{IMeshComponent}. Finally, the class \arcanemat{ComponentCellVector}
allows creating a vector of \arcanemat{ComponentCell} and can be used as an
\arcanemat{EnvCellVector}
or \arcanemat{MatCellVector}.

The method \arcanemat{ComponentCell::superCell()} returns the
\arcanemat{ComponentCell} of the immediately higher hierarchical level. It is
also possible
to iterate over the sub-cells of an \arcanemat{ComponentCell} via the macro
ENUMERATE_CELL_COMPONENTCELL():

\snippet MeshMaterialTesterModule_Samples.cc SampleComponentSuperItem

Finally, there is a macro ENUMERATE_COMPONENT() which allows iterating
over a list of components, and thus can replace ENUMERATE_MAT() or
ENUMERATE_ENV().

## Material Variables {#arcanedoc_materials_manage_variable}

It is also possible to declare variables only on environments that do not have
values on materials. For more information, see section
\ref arcanedoc_execution_env_variables.

Material variables are similar to mesh variables but in addition to having a
value on classical cells, they have a value per material and per environment
present in the cell. For a cell that has 3 environments, with 2 materials for
environment 1, 3 for environment 2, and 5 for environment 3, the number of
values is therefore 14 (10 for materials, 3 for environments, and 1 for the
global value). The values per materials and per environments are called partial
values.

Currently, material variables are only available on cells. The base class
managing these variables is
MeshMaterialVariableRef. The possible types are as follows and are defined in
the file \c "MeshMaterialVariableRef.h".

<table>
<tr><th>Name</th><th>Description</th></th>
<tr><td>\arcanemat{MaterialVariableCellByte}</td><td>material variable of type \arcane{Byte}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellReal}</td><td>material variable of type \arcane{Real}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellInt16}</td><td>material variable of type \arcane{Int16}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellInt32}</td><td>material variable of type \arcane{Int32}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellInt64}</td><td>material variable of type \arcane{Int64}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellReal2}</td><td>material variable of type \arcane{Real2}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellReal3}</td><td>material variable of type \arcane{Real3}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellReal2x2}</td><td>material variable of type \arcane{Real2x2}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellReal3x3}</td><td>material variable of type \arcane{Real3x3}</td></tr>
</table>

It is also possible to define array variables on materials, which have the
following type:

<table>
<tr><th>Name</th><th>Description</th></th>
<tr><td>\arcanemat{MaterialVariableCellArrayByte}</td><td>material variable of type array of \arcane{Byte}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayReal}</td><td>material variable of type array of \arcane{Real}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayInt16}</td><td>material variable of type array of \arcane{Int16}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayInt32}</td><td>material variable of type array of \arcane{Int32}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayInt64}</td><td>material variable of type array of \arcane{Int64}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayReal2}</td><td>material variable of type array of \arcane{Real2}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayReal3}</td><td>material variable of type array of \arcane{Real3}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayReal2x2}</td><td>material variable of type array of \arcane{Real2x2}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayReal3x3}</td><td>material variable of type array of \arcane{Real3x3}</td></tr>
</table>

\note For information, internally in Arcane, these partial values are managed as
a classic array variable, but this implementation may evolve. Since version 2.0
of %Arcane, in order to distinguish these variables from classical variables,
they are tagged with the value "Material". For example, the command

```cpp
Arcane::IVariable* var = ...;
if (var->hasTag("Material")){
  // This is a partial value.
}
```

For now, it is only possible to create scalar variables on cells, with one of
the following data types: #Real, #Int32, #Int64, Real2, Real3, Real2x2, or
Real3x3. The name of the corresponding class is the same as for classical
variables, but prefixed with 'Material'. For example, for a Real3 variable, the
name is \a MaterialVariableCellReal3.

### Construction {#arcanedoc_materials_manage_variable_build}

The declaration of material variables is done in a manner similar to that of
mesh variables. It is also possible to declare them in the axl file, in the same
way as a classical variable, by adding the attribute \c material with the value
\c true. The valid values for \c dimension are \c 0 or
\c 1.

```xml
<variable field-name="mat_density"
         name="Density"
         data-type="real"
         item-kind="cell"
         dim="0"
         material="true"
/>
```

The construction is done with an object of type MaterialVariableBuildInfo which
references the corresponding IMeshMaterialMng or in the same way as a classical
variable, via the VariableBuildInfo. For example:

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialCreateVariable

\note The construction of material variables is thread-safe.

As with classic variables, the previous instructions create a variable only if
none with the same name exists. Otherwise, they retrieve a reference to the
corresponding already created variable.

### Usage {#arcanedoc_materials_manage_variable_usage}

To access a material variable's value, simply use the [] operator with one of
the following types as an argument: ComponentCell, Cell, MatCell, EnvCell, or
AllEnvCell.

```cpp
Arcane::Cell global_cell;
Arcane::Materials::MatCell mat_cell;
Arcane::Materials::EnvCell env_cell;
mat_density[global_cell]; // Global value
mat_density[mat_cell];    // Value for a material
mat_density[env_cell];    // Value for an environment.
```

The global value is shared with that of standard Arcane variables of the same
name. For example:

```cpp
Arcane::Materials::IMeshMaterialMng* material_mng = ...;
Arcane::Materials::MaterialVariableBuildInfo mat_info(material_mng,"Density"))
Arcane::Materials::MaterialVariableCellReal mat_density(mat_info);
Arcane::VariableBuildInfo info(defaultMesh(),"Density"))
Arcane::VariableCellReal density(info);

mat_density[global_cell] = 3.0;
info() << density[global_cell]; // Displays 3.0
```

It is also possible to retrieve the global variable associated with a material
variable via the globalVariable() method:

```cpp
Arcane::Materials::IMeshMaterialMng* material_mng = ...;
Arcane::Materials::MaterialVariableCellReal mat_density(Arcane::Materials::MaterialVariableBuildInfo(material_mng,"Density"));
Arcane::VariableCellReal& density(mat_density.globalVariable());
```

The implementation of material variables aims to limit memory usage. With this
goal in mind, values in environments and materials can use the same memory area,
and in this case, modifying the environment value also modifies the material
value and vice versa. This is the case if the following conditions are met in a
cell \a cell:
- only one material (MAT) in an environment (ENV), then var[MAT]==var[ENV]
- only one environment (ENV) in the cell, then var[ENV] == var[cell]

### Synchronizations {#arcanedoc_materials_manage_variable_synchronize}

It is possible to synchronize the values of material variables, just like
classic variables, using the \arcanemat{MeshMaterialVariableRef::synchronize()}
function.

\warning Attention, it is still necessary that the information about the present
materials is consistent across all subdomains: if a cell exists in multiple
subdomains, it must have the same materials and environments in each subdomain.

It is possible to guarantee that all information about materials and
environments is consistent across subdomains by calling the
\arcanemat{IMeshMaterialMng::synchronizeMaterialsInCells()} method. It is also
possible to check this consistency by calling
\arcanemat{IMeshMaterialMng::checkMaterialsInCells()}.

Since version 2.3.7, there are several implementations for synchronization. The
default version is not optimized and performs synchronization per material
registered in \arcanemat{IMeshMaterialMng}. The recommended optimized version is
version 6. To use it, you must call the
\arcanemat{IMeshMaterialMng::setSynchronizeVariableVersion()} method with the
value 6. It is also possible to
perform multiple synchronizations at once using the
\arcanemat{MeshMaterialVariableSynchronizerList} class. This allows optimizing
communications by reducing the number of messages between processors.
For example:
```cpp
Arcane::Materials::IMeshMaterialMng* material_mng = ...;
Arcane::Materials::MaterialVariableCellReal temperature = ...;
Arcane::Materials::MaterialVariableCellInt32 mat_index = ...;
Arcane::Materials::MaterialVariableCellReal pressure = ...;
// Creation of the list of variables to synchronize.
Arcane::Materials::MeshMaterialVariableSynchronizerList mmvsl(material_mng);

// Adds 3 variables to the list.
temperatture.synchronize(mmvsl);
mat_index.synchronize(mmvsl);
pressure.synchronize(mmvsl);

// Executes the synchronization for the 3 variables at once.
mmvsl.apply();
```

Since version 3.6, there are two other versions of synchronization that function
identically to version 6 but with the following modifications:
- Version 7 performs only one memory allocation for the send and receive
  buffers (whereas version 6 performs as many allocations as there are
  neighboring subdomains)
- Version 8, which is identical to version 7 in detail, keeps the allocated
  buffers between two synchronizations. This version avoids successive
  allocations/deallocations at the cost of increased memory usage.
  These two versions aim to avoid too many allocation/deallocation cycles, which
  can demand a significant workload for network cards for direct memory access
  (RMA).

### Dependency Management {#arcanedoc_materials_manage_variable_dependencies}

It is possible to use the dependency mechanism on material variables. This
mechanism is similar to that of classic variables but allows managing
dependencies per material.

\note Unlike dependencies on classic variables, dependencies on materials do not
manage physical time, and it is not possible, for example, to make dependencies
on the previous physical time (by specifying
\arcane{IVariable::DPT_PreviousTime}, for example).

The operation works as follows:

\snippet MeshMaterialTesterModule_Samples.cc SampleDependencies

When calling \arcanemat{MeshMaterialVariableRef::update()}, it is necessary to
specify the material on which the update should be performed as an argument. The
calculation method must therefore have a material as an argument, and at the end
of the calculation, call \arcanemat{MeshMaterialVariableRef::setUpToDate(mat)}
with \a mat being the recalculated material.
For example:

\snippet MeshMaterialTesterModule_Samples.cc SampleDependenciesComputeFunction

## Environment Variables {#arcanedoc_materials_manage_environment_variable}

It is also possible to define variables that only have partial values in
environments and not in materials. Apart from this difference, they behave like
material variables. The available environment variables are as follows:

<table>
<tr><th>Name</th><th>Description</th></th>
<tr><td>\arcanemat{EnvironmentVariableCellByte}</td><td>environment variable of type \arcane{Byte}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellReal}</td><td>environment variable of type \arcane{Real}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellInt16}</td><td>environment variable of type \arcane{Int16}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellInt32}</td><td>environment variable of type \arcane{Int32}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellInt64}</td><td>environment variable of type \arcane{Int64}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellReal2}</td><td>environment variable of type \arcane{Real2}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellReal3}</td><td>environment variable of type \arcane{Real3}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellReal2x2}</td><td>environment variable of type \arcane{Real2x2}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellReal3x3}</td><td>environment variable of type \arcane{Real3x3}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayByte}</td><td>environment variable of type array of \arcane{Byte}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayReal}</td><td>environment variable of type array of \arcane{Real}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayInt16}</td><td>environment variable of type array of \arcane{Int16}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayInt32}</td><td>environment variable of type array of \arcane{Int32}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayInt64}</td><td>environment variable of type array of \arcane{Int64}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayReal2}</td><td>environment variable of type array of \arcane{Real2}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayReal3}</td><td>environment variable of type array of \arcane{Real3}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayReal2x2}</td><td>environment variable of type array of \arcane{Real2x2}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayReal3x3}</td><td>environment variable of type array of \arcane{Real3x3}</td></tr>
</table>

In the axl file, these variables can be defined by specifying the \c environment
attribute to \c true.

```xml
<variable field-name="mat_density"
         name="Density"
         data-type="real"
         item-kind="cell"
         dim="0"
         environment="true"
/>
```

\warning Since the internal structures of material and environment cells are
unified, it is possible at compilation time to index an environment variable
with a \arcanemat{MatCell}. Since there are no associated material values, this
will cause an invalid memory access which is highly likely to result in a
segmentation fault (SegmentationFault). These errors are detected in CHECK mode,
so it is preferable to use this mode for development.

## Parallelization of Loops on Materials and Environments {#arcanedoc_materials_manage_concurrency}

Just like loops on entities (see \ref arcanedoc_parallel_concurrency), it is
possible to execute loops on environments or materials in parallel. This is done
similarly to loops on entities, using the \arcane{Parallel::Foreach} method. For
example:

\snippet MeshMaterialTesterModule_Samples.cc SampleConcurrency

## Optimization of Modifications on Materials and Environments. {#arcanedoc_materials_manage_optimization}

Modification of material and environment cells is done via the
\arcanemat{MeshMaterialModifier} class. This modification is done through
successive calls to \arcanemat{MeshMaterialModifier::addCells()} or
\arcanemat{MeshMaterialModifier::removeCells()}. These methods allow recording
the list of modifications, but they are only actually executed upon the
destruction of the \arcanemat{MeshMaterialModifier} instance.

By default, the behavior is as follows:
- saving the values of all material variables
- applying the modifications (which consists only of modifying the list of
  entities of the cell groups associated with materials and environments)
- restoring the values of all material variables. During restoration, if a new
  material cell is created, its value depends on
  \arcanemat{IMeshMaterialMng::isDataInitialisationWithZero()}.

Saving and restoring values are CPU and memory-intensive operations. It is
possible to disable these operations via setKeepValuesAfterChange(), but of
course, in this case, partial values are not preserved.

To optimize these material modifications, it is possible to bypass these
save/restore operations. To do this, you must use the
IMeshMaterialMng::setModificationFlags(int v) method. This method must be called
before \arcanemat{IMeshMaterialMng::endCreate()}.
The argument used is a combination of bits from the
\arcanemat{eModificationFlags} enumeration:
- \arcanemat{eModificationFlags::GenericOptimize}: indicates that you wish to
  enable optimizations.
- \arcanemat{eModificationFlags::OptimizeMultiAddRemove}: indicates that you
  activate optimizations in the case where there are multiple additions or
  removals with the same MeshMaterialModifier.
- \arcanemat{eModificationFlags::OptimizeMultiMaterialPerEnvironment} indicates
  that you activate optimizations in the case where there are multiple materials
  in the environment.

\warning The value
\arcanemat{eModificationFlags::OptimizeMultiMaterialPerEnvironment} is only
available from version 2.3.2 of %Arcane. On earlier versions, no optimization
is performed if one of the modified materials is not the only material in the
environment.

For example, suppose the three series of modifications:
```cpp
{
  Arcane::Materials::MeshMaterialModifier m1(m_material_mng);
  m1.addCells(mat1,ids);
}
{
  Arcane::Materials::MeshMaterialModifier m2(m_material_mng);
  m2.addCells(mat1,ids1);
  m2.addCells(mat2,ids2);
  m2.removeCells(mat1,ids3);
}
{
  Arcane::Materials::MeshMaterialModifier m3(m_material_mng);
  m3.removeCells(mat2,ids);
}
```

Depending on the values specified during initialization, you will have:

```cpp
int flags1 = (int)Arcane::Materials::eModificationFlags::GenericOptimize;
m_material_mng->setModificationFlags(flags1);
// Only m1 and m3 are optimized.

int flags2 = (int)Arcane::Materials::eModificationFlags::GenericOptimize | (int)Arcane::Materials::eModificationFlags::OptimizeMultiAddRemove;
m_material_mng->setModificationFlags(flags2);
// m1, m2 and m3 are optimized.
```

It is possible to override the used optimizations via the environment variable
ARCANE_MATERIAL_MODIFICATION_FLAGS. This variable must contain an integer value
corresponding to the one used as an argument for
\arcanemat{IMeshMaterialMng::setModificationFlags()} (namely 1 for general
optimization, 3 to further optimize multiple additions/removals, and 7 to also
optimize multi-material environments).

### Notes on Implementation {#arcanedoc_materials_manage_optimization_implementation}

\htmlonly
<span style='background: red'>IMPORTANT</span>
\endhtmlonly

<h4>NOTE 1</h4>

Currently, the optimized methods do not reuse partial values when cells are
deleted and then added to the same material, which leads to a gradual increase
in the memory used by partial values. However, it is possible to free up this
extra memory by calling IMeshMaterialMng::forceRecompute().

<h4>NOTE 2</h4>

The behavior in optimized mode when there is deletion followed by addition of
the same cell in the same material is different from classic mode. For example:

```cpp
MeshMaterialModifier m1(m_material_mng);
Array<Int32> add_ids;
Array<Int32> remove_ids;

remove_ids.add(5);
remove_ids.add(9);
add_ids.add(5);
add_ids.add(7);

m1.removeCells(mat1,remove_ids);
m1.addCells(mat1,add_ids);
```

The cell with localId() \a 5 is first deleted and then put back into the
material. In classic mode, the cell value will be the same as before the
modification because the value is restored from the save. In optimized mode,
with eModificationFlags::OptimizeMultiAddRemove specified, the cell is first
removed from the material and then recreated. Its value will therefore be that
of a newly created material cell, so 0 if
IMeshMaterialMng::isDataInitialisationWithZero() or the value of the associated
global cell otherwise.

<h4>NOTE 3</h4>

Finally, the optimized methods are stricter than the classic methods, and
certain operations that were tolerated in classic mode are no longer tolerated:
- specifying in the list of cells to add a cell that is already in the
  material.
- specifying in the list of cells to remove a cell that is not in the material.
- specifying the same cell multiple times in the list of cells to remove or
  add.

```cpp
MeshMaterialModifier mm(m_material_mng);
Int32Array ids;
ids.add(5);
mm.addCells(mat1,ids); // Error if mesh 5 is already in mat1
mm.removeCells(mat1,ids); // Error if mesh 5 is not in mat1
ids.add(6);
ids.add(6); // Error if \a ids contains the same cells multiple times.
```

If one of the previous cases occurs, there is a high chance that the code will
crash. To prevent this, CHECK mode detects these errors, signals them in the
listing, and filters them so that it works correctly. These errors are signaled
in the listing with messages such as the following:

\verbatim
ERROR: item ... is present several times in add/remove list for material mat=...
ERROR: item ... is already in material mat=...
ERROR: item ... is not in material mat=...
\endverbatim

In release mode, detection and correction only happen if the ARCANE_CHECK
environment variable is set and equals 1.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_materials
</span>
<span class="next_section_button">
\ref arcanedoc_materials_loop
</span>
</div>
