# Main Concepts {#arcanedoc_glossary_terms}

[TOC]

This page defines the main concepts of the %Arcane platform. The terms are
grouped into the core concepts (subdomain, mesh, Cartesian mesh, mesh item,
item family, item group, parallel manager, variable, variable manager, module
and service) and the materials-system concepts (material manager, material,
constituent, environment and block). Each term links to the class that
implements it.

## Subdomain {#arcanedoc_glossary_terms_subdomain}

A **subdomain** (\arcane{ISubDomain}) is the basic unit of execution of an
%Arcane application. It represents the part of the computation executed by one
thread (or by one group of threads) in the hybrid (MPI + threads) execution of
%Arcane. A subdomain owns its own meshes, variables, modules and services, and
is the object from which all these resources are accessed. In a sequential
execution there is a single subdomain; in a parallel execution there is one
subdomain per MPI rank (and possibly more when threads are used). A subdomain
is associated with a sub-partition of a parallel manager
(\arcane{IParallelMng}).

## Mesh {#arcanedoc_glossary_terms_mesh}

A **mesh** (\arcane{IMesh}) is a set of mesh entities (nodes, edges, faces and
cells) together with their connectivity. A mesh is owned by a subdomain and,
when created with a non-sequential \arcane{IParallelMng}, is distributed
across the ranks: each rank holds a local part of the mesh plus one or more
ghost layers of neighbouring entities. A mesh has a spatial dimension (2 or 3)
and exposes one family per entity kind through its node, edge, face and cell
families. A mesh can be created empty and then filled, or read directly from a
file.

## Cartesian Mesh {#arcanedoc_glossary_terms_cartesianmesh}

A **Cartesian mesh** (\arcane{ICartesianMesh}) is a special kind of mesh whose
cells are arranged on a regular structured grid. It is associated with a
standard \arcane{IMesh} (retrieved or created from it via
\arcane{ICartesianMesh::getReference()}) and adds, on top of the regular mesh
facilities, directional access to the entities (cells, faces and nodes in a
given direction) and the operations needed for adaptive mesh refinement
(refining or coarsening blocks of the mesh). A Cartesian mesh is typically used
by codes that solve the equations on a structured grid.

## Mesh Item {#arcanedoc_glossary_terms_item}

A **mesh item** (or simply an *item*, \arcane{Item}) is a single entity of a
mesh: one node, one edge, one face or one cell. The `Item` class is the base
class of the typed entities (`Node`, `Edge`, `Face`, `Cell` and `Particle`).
An item has a local id (unique within the local part of its family) and a
unique id (globally unique, used to identify the entity across ranks), and it
belongs to an item family. An item is a lightweight handle: the associated data
(coordinates, variable values) are stored in the family.

## Item Family {#arcanedoc_glossary_terms_itemfamily}

An **item family** (\arcane{IItemFamily}) is the collection of all the items
of a given kind (nodes, edges, faces or cells) of a mesh. A mesh exposes its
families through \arcane{IMeshBase::nodeFamily()},
\arcane{IMeshBase::edgeFamily()}, \arcane{IMeshBase::faceFamily()} and
\arcane{IMeshBase::cellFamily()}. A family stores the items, their
connectivity and the variables defined on them, and is the primary object used
to iterate over and to modify mesh entities. Particles form a separate kind of
item family (\arcane{IParticleFamily}).

## Item Group {#arcanedoc_glossary_terms_itemgroup}

An **item group** (\arcane{ItemGroup}) is a subset (possibly empty) of the
items of a given item family. Unlike an item family, which is the whole set of
the items of a kind in a mesh, a group is an arbitrary subset that can be
created, combined (union, intersection, difference) and iterated independently
of the family. A group is created through its family using
\arcane{IItemFamily::createGroup()} or \arcane{IItemFamily::findGroup()}.
Groups are used to select specific sets of items (for example the cells of a
material or of a region) on which to perform operations.

## Parallel Manager {#arcanedoc_glossary_terms_parallelmng}

A **parallel manager** (\arcane{IParallelMng}) is the object that manages the
parallel decomposition (MPI ranks and/or threads) of an %Arcane application and
provides the collective communication operations (barriers, reductions,
exchanges) used by the framework and by the application. Each subdomain is
associated with a sub-partition of a parallel manager. The default parallel
manager of the application covers all the ranks, and sub-managers can be
created for subsets of ranks. A sequential parallel manager (a single rank) is
used for non-parallel meshes.

## Variable {#arcanedoc_glossary_terms_variable}

A **variable** (\arcane{IVariable}) is a named piece of data associated with a
subdomain. There are two main kinds of variables: *common variables*, which
hold a single value per subdomain, and *item variables*, which hold one value
per mesh item (for example one value per cell). A variable is declared either
in the AXL descriptor of a module or of a service, or directly in the code, and
is accessed in the code through a variable reference (\arcane{VariableRef} and
its derived classes, the typed variable references).

## Variable Manager {#arcanedoc_glossary_terms_variablemng}

The **variable manager** (\arcane{IVariableMng}) is the object that maintains
the list of the variables declared in a subdomain. It allows these variables to
be retrieved and to be read or written (for example for checkpointing or for
post-processing). It is obtained from the subdomain via
\arcane{ISubDomain::variableMng()}.

## Module {#arcanedoc_glossary_terms_module}

A **module** (\arcane{IModule}) is a basic unit of application code in %Arcane.
A module groups a set of *entry points* (the points from which the %Arcane time
loop calls the module), *variables* and *configuration options*. A module is
represented by a class and by an XML file called the *module descriptor*
(extension `.axl`) that describes its variables, entry points and options.
Module instances are created and configured by the %Arcane runtime from the
simulation case file.

## Service {#arcanedoc_glossary_terms_service}

A **service** (\arcane{IService}) has the same characteristics as a module,
except that it has no entry point. A service is used to factorize code across
several modules (for example a numerical scheme or an equation of state shared
by several physics modules) or to parameterize a module with different
algorithms. Like a module, a service is represented by a class and by an XML
*service descriptor* (extension `.axl`) that describes the interface it
implements, its options and its variables. A service instance is created by the
%Arcane runtime when a module references it.

## Material Manager {#arcanedoc_glossary_terms_materialmng}

\note The materials-system concepts (material manager, material, constituent,
environment and block) are optional: it is possible to use %Arcane without
using any of them.

A **material manager** (\arcanemat{IMeshMaterialMng}) is the object that
manages all the materials, environments and blocks of a mesh, as well as their
associated material and environment variables. There is one material manager
per mesh, retrieved or created via the static method
\arcanemat{IMeshMaterialMng::getReference()}. The materials and environments
are registered and created during the initialization of the computation and
cannot be modified afterward; only the list of cells of a material or an
environment can change during the run (via a \arcanemat{MeshMaterialModifier}).

## Material {#arcanedoc_glossary_terms_material}

A **material** (\arcanemat{IMeshMaterial}) is a named set of mesh cells that
represents a physical substance present in those cells. It belongs to the
materials system (namespace `Arcane::Materials`). A material is associated only
with cells, and its list of cells is dynamic: it can change during the
computation. A material belongs to one or more environments. The materials of a
mesh are managed by an \arcanemat{IMeshMaterialMng} instance (one per mesh).

## Constituent {#arcanedoc_glossary_terms_constituent}

A **constituent** (\arcanemat{ConstituentItem}) is the occurrence of a
component (a material or an environment) inside a single cell. A cell can
contain several materials and several environments, and the constituents of a
cell form a hierarchy: \arcanemat{AllEnvCell} (the cell together with all its
environments), \arcanemat{EnvCell} (one environment of the cell) and
\arcanemat{MatCell} (one material of the cell). Constituent items are used to
access the per-cell data and variables of the materials and environments.

## Environment {#arcanedoc_glossary_terms_environment}

An **environment** (\arcanemat{IMeshEnvironment}) is a named grouping of one
or more materials of a mesh. It belongs to the materials system (namespace
`Arcane::Materials`) and forms an intermediate level between the (optional)
blocks and the materials. The list of cells of an environment is the union of
the lists of cells of its materials and is recomputed automatically when the
materials change. As for materials, an environment is associated only with
cells and can be empty.

## Block {#arcanedoc_glossary_terms_block}

A **block** (\arcanemat{IMeshBlock}) is an optional named grouping, located at
a level above the environments, that groups a set of environments together with
a group of cells. It belongs to the materials system (namespace
`Arcane::Materials`) and is managed by the \arcanemat{IMeshMaterialMng}
instance along with the materials and the environments.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_glossary
</span>
</div>
