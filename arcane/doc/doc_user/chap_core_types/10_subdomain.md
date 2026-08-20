# The SubDomain {#arcanedoc_core_types_subdomain}

[TOC]

The \arcane{ISubDomain} interface (defined in `arcane/core/ISubDomain.h`) is
the interface of the *subdomain manager*. It is the central object representing
the local state and the execution environment of a single computation unit of
an %Arcane application.

## What is a Subdomain ? {#arcanedoc_core_types_subdomain_what}

When an %Arcane application runs in parallel, the global mesh is partitioned
into several *subdomains*: each computation unit (an MPI rank, possibly
associated with a set of threads) owns a portion of the mesh together with the
local variables, modules, services, and case options associated with it.
The \arcane{ISubDomain} instance is the handle to this local portion of the
application.

The object hierarchy is as follows:

```
IApplication   (one per executable)
└── ISession   (one per computation session)
    └── ISubDomain   (one per (rank, thread) unit)
        ├── IVariableMng, IModuleMng, IEntryPointMng, IServiceMng, ...
        ├── IMeshMng → IMesh (default mesh)
        ├── ICaseMng (the .arc data set)
        └── ...
```

- \arcane{IApplication} is the process-level manager: it is shared by all the
  subdomains of the process (main factory, application information, resource
  manager, ...).
- \arcane{ISession} groups one or more subdomains. In *domain replication*
  (several independent computations sharing the same processes), a session
  contains several \arcane{ISubDomain} instances, one per replica.
- \arcane{ISubDomain} is where the identity of the computation unit lives:
  `subDomainId()` is the rank in the parallel group of the subdomain and
  `nbSubDomain()` is the size of this group.

In domain replication, `parallelMng()` covers only the parallel group of one
replica, while `allReplicaParallelMng()` covers the group of *all* the
replicas. Without replication, both managers are the same.

## Access to the Main Managers {#arcanedoc_core_types_subdomain_managers}

Most of the \arcane{ISubDomain} interface is made of accessors to the managers
owned by the subdomain. They are grouped here by category:

| Category | Accessors |
|---|---|
| Data | `variableMng()` (all the \arcane{IVariable} instances), `meshMng()` and `defaultMesh()` / `defaultMeshHandle()` / `meshes()`, `propertyMng()` |
| Structure | `moduleMng()`, `entryPointMng()`, `moduleMaster()` (the master module, always created) |
| Parallelism / threads | `parallelMng()`, `allReplicaParallelMng()`, `threadMng()`, `timerMng()`, `timeStats()`, `loadBalanceMng()` |
| Time | `timeLoopMng()`, `timeHistoryMng()` |
| I/O and case file | `ioMng()`, `caseMng()` (the \arcane{ICaseMng} of the data set), `caseDocument()`, `caseOptionsMain()`, `commonVariables()` |
| Protection | `checkpointMng()` |
| Others | `acceleratorMng()`, `memoryInfo()`, `physicalUnitSystem()`, `configuration()`, `mainFactory()`, `session()`, `application()`, `applicationInfo()` |

\arcane{ISubDomain} is therefore the place to look for anything that is "mine"
in the application: *my* variables, *my* mesh, *my* parallel group, *my* case
options. For a detailed description of the mesh and of the entity families it
contains, refer to \ref arcanedoc_core_types_mesh and
\ref arcanedoc_core_types_item_family.

## Mesh-related Methods {#arcanedoc_core_types_subdomain_mesh}

The subdomain manages the meshes associated with the computation unit. The
main methods are:

- `defaultMesh()` / `defaultMeshHandle()`: the default mesh of the subdomain
  (named `"Mesh0"`). The handle always exists, even if the associated mesh has
  not yet been created.
- `meshes()`: the list of all the meshes of the subdomain.
- `allocateMeshes()`: allocates the \arcane{IMesh} instances (they are created
  but contain no entities). This method must be called before any other
  operation involving the mesh.
- `readOrReloadMeshes()`: reads the meshes. At startup, the meshes are read
  from the data set; during a restart, they are loaded from a protection.
- `initializeMeshVariablesFromCaseFile()`: initializes the variables whose
  values are specified in the data set.
- `doInitMeshPartition()`: applies the initial mesh partitioning (see
  \ref arcanedoc_parallel_intro).
- `setInitialPartitioner(IInitialPartitioner*)`: sets the partitioner to use
  for the initial partition. The subdomain takes ownership of the instance and
  destroys it at the end of the computation. This method must be called before
  the module initialization, for example in a *construction* entry point.

\warning The methods `mesh()`, `findMesh()`, `addMesh()` and `meshDimension()`
are deprecated. Use `defaultMeshHandle()`, `meshMng()->findMeshHandle()`,
`meshMng()->meshFactoryMng()` and `mesh()->dimension()` instead.

## Case File and Directories {#arcanedoc_core_types_subdomain_case}

The subdomain keeps track of the data set (the `.arc` file) and of the output
directories:

- `caseName()`, `caseFullFileName()`, `fillCaseBytes()`, `setCaseName()`:
  information on the data set. `setCaseName()` must be called before
  initialization.
- `caseDocument()`: the XML document of the data set.
- `caseOptionsMain()`: the global options of the data set
  (\arcane{CaseOptionsMain}).
- `commonVariables()`: the "standard" variables (time, time step, ...).

Three output directories can be set (the directories must exist and the methods
must be called before initialization):

- `exportDirectory()` / `setExportDirectory()`: the base directory for
  exports (protections and restarts).
- `storageDirectory()` / `setStorageDirectory()`: the base directory for
  exports requiring archiving. If not set, `exportDirectory()` is used.
- `listingDirectory()` / `setListingDirectory()`: the base directory for
  listings (logs, execution information).

## Creation and Lifecycle {#arcanedoc_core_types_subdomain_lifecycle}

A subdomain is created from a \arcane{SubDomainBuildInfo} object, which
provides:
- the parallel manager of the subdomain (`parallelMng()`),
- the subdomain index in the session (`index()`),
- the data set: file name (informative) and content (`caseBytes()`),
- optionally, the parallel manager of all the replicas
  (`allReplicaParallelMng()`) for domain replication.

The creation goes through \arcane{ISession}:

```cpp
SubDomainBuildInfo sdbi(parallel_mng, index);
sdbi.setCaseFileName(case_file_name);
ISubDomain* sub_domain = session->createSubDomain(sdbi);
```

The typical lifecycle of a subdomain is:

| Phase | Methods | Description |
|---|---|---|
| Build | (internal) | Creation of all the subdomain managers (variable manager, module manager, entry point manager, service manager, case manager, checkpoint manager, time loop manager, ...) and of the default mesh handle. |
| Initialize | (internal) | Initialization of the accelerator, reading of the data set (validates the XML and merges the configuration), creation of the default mesh, loading of the subdomain services, creation of the master module. |
| Mesh setup | `allocateMeshes()`, `readOrReloadMeshes()`, `initializeMeshVariablesFromCaseFile()`, `doInitMeshPartition()` | Allocation and reading of the meshes, initialization of the mesh variables from the data set, application of the initial partition. |
| Restart support | `setIsContinue()` / `isContinue()` | Flag indicating that a restart is being performed. `setIsContinue()` must be called before `allocateMeshes()` so that the meshes are loaded from a protection instead of the data set. |
| Run | (application code) | The application entry points are executed through the time loop. |
| Destroy | `doExitModules()`, `destroy()` | Execution of the *exit* entry points, then destruction of all the subdomain objects (the variable manager is destroyed last, since every object may contain variables). |

`isInitialized()` / `setIsInitialized()` indicate whether the subdomain has
been initialized. `onDestroyObservable()` allows registering a notification
that is called before the subdomain is destroyed.

## Getting the SubDomain from the Code {#arcanedoc_core_types_subdomain_get}

Nearly every object owned by the subdomain provides a `subDomain()` accessor to
climb back to it:

- services: `subDomain()` inherited from the service base class,
- modules: `subDomain()` inherited from \arcane{IModule},
- entry points: `subDomain()` inherited from \arcane{IEntryPoint},
- meshes: \arcane{IMesh::subDomain}(),
- variables: \arcane{IVariable::subDomain}(),
- item families: \arcane{IItemFamily::subDomain}(),
- case options: \arcane{ICaseOptions::subDomain}().

The canonical pattern inside an entry point or a service method is therefore:

```cpp
void MyModule::onDoSomething()
{
  IVariableMng* vpm = subDomain()->variableMng();
  IMesh* mesh = subDomain()->defaultMesh();
  IParallelMng* pm = subDomain()->parallelMng();
  // ...
}
```

## Key Points {#arcanedoc_core_types_subdomain_notes}

- \arcane{ISubDomain} is the per-replica root object of the application:
  anything that needs the local variables, the local mesh, the local parallel
  group, or the local case options goes through it.
- It is *not* the application: `application()` gives the process-wide objects.
  The subdomain is where the identity of the computation unit lives
  (`subDomainId()` is the rank).
- The ordering of the initialization methods matters: `setCaseName()`,
  `setExportDirectory()`, `setInitialPartitioner()` before initialization;
  `setIsContinue()` before `allocateMeshes()`.
- The subdomain is the seam for the advanced execution modes: domain
  replication (several subdomains per session), restarts (protections), and
  direct execution (tests), which all plug in through its setters
  (`directExecution()` / `setDirectExecution()`).

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_array_views
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_mesh
</span>
</div>
