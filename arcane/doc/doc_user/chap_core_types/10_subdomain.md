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
  \arcane{ISubDomain::subDomainId()} is the rank in the parallel group of the
  subdomain and \arcane{ISubDomain::nbSubDomain()} is the size of this group.

In domain replication, \arcane{ISubDomain::parallelMng()} covers only the
parallel group of one replica, while \arcane{ISubDomain::allReplicaParallelMng()}
covers the group of *all* the replicas. Without replication, both managers are
the same.

## Access to the Main Managers {#arcanedoc_core_types_subdomain_managers}

Most of the \arcane{ISubDomain} interface is made of accessors to the managers
owned by the subdomain. They are grouped here by category:

| Category | Accessors |
|---|---|
| Data | \arcane{ISubDomain::variableMng()} (all the \arcane{IVariable} instances), \arcane{ISubDomain::meshMng()} and \arcane{ISubDomain::defaultMesh()} / \arcane{ISubDomain::defaultMeshHandle()} / \arcane{ISubDomain::meshes()}, \arcane{ISubDomain::propertyMng()} |
| Structure | \arcane{ISubDomain::moduleMng()}, \arcane{ISubDomain::entryPointMng()}, \arcane{ISubDomain::moduleMaster()} (the master module, always created) |
| Parallelism / threads | \arcane{ISubDomain::parallelMng()}, \arcane{ISubDomain::allReplicaParallelMng()}, \arcane{ISubDomain::threadMng()}, \arcane{ISubDomain::timerMng()}, \arcane{ISubDomain::timeStats()}, \arcane{ISubDomain::loadBalanceMng()} |
| Time | \arcane{ISubDomain::timeLoopMng()}, \arcane{ISubDomain::timeHistoryMng()} |
| I/O and case file | \arcane{ISubDomain::ioMng()}, \arcane{ISubDomain::caseMng()} (the \arcane{ICaseMng} of the data set), \arcane{ISubDomain::caseDocument()}, \arcane{ISubDomain::caseOptionsMain()}, \arcane{ISubDomain::commonVariables()} |
| Protection | \arcane{ISubDomain::checkpointMng()} |
| Others | \arcane{ISubDomain::acceleratorMng()}, \arcane{ISubDomain::memoryInfo()}, \arcane{ISubDomain::physicalUnitSystem()}, \arcane{ISubDomain::configuration()}, \arcane{ISubDomain::mainFactory()}, \arcane{ISubDomain::session()}, \arcane{ISubDomain::application()}, \arcane{ISubDomain::applicationInfo()} |

\arcane{ISubDomain} is therefore the place to look for anything that is "mine"
in the application: *my* variables, *my* mesh, *my* parallel group, *my* case
options. For a detailed description of the mesh and of the entity families it
contains, refer to \ref arcanedoc_core_types_mesh and
\ref arcanedoc_core_types_item_family.

## Mesh-related Methods {#arcanedoc_core_types_subdomain_mesh}

The subdomain manages the meshes associated with the computation unit. The
main methods are:

- \arcane{ISubDomain::defaultMesh()} / \arcane{ISubDomain::defaultMeshHandle()}:
  the default mesh of the subdomain (named `"Mesh0"`). The handle always
  exists, even if the associated mesh has not yet been created.
- \arcane{ISubDomain::meshes()}: the list of all the meshes of the subdomain.
- \arcane{ISubDomain::allocateMeshes()}: allocates the \arcane{IMesh}
  instances (they are created but contain no entities). This method must be
  called before any other operation involving the mesh.
- \arcane{ISubDomain::readOrReloadMeshes()}: reads the meshes. At startup, the
  meshes are read from the data set; during a restart, they are loaded from a
  protection.
- \arcane{ISubDomain::initializeMeshVariablesFromCaseFile()}: initializes the
  variables whose values are specified in the data set.
- \arcane{ISubDomain::doInitMeshPartition()}: applies the initial mesh
  partitioning (see \ref arcanedoc_parallel_intro).
- \arcane{ISubDomain::setInitialPartitioner()}: sets the partitioner to use
  for the initial partition. The subdomain takes ownership of the instance and
  destroys it at the end of the computation. This method must be called before
  the module initialization, for example in a *construction* entry point.

\warning The methods \arcane{ISubDomain::mesh()}, `findMesh()`, `addMesh()` and
\arcane{ISubDomain::meshDimension()} are deprecated. Use
\arcane{ISubDomain::defaultMeshHandle()}, \arcane{IMeshMng::findMeshHandle()},
\arcane{IMeshMng::meshFactoryMng()} and \arcane{IMesh::dimension()} instead.

## Case File and Directories {#arcanedoc_core_types_subdomain_case}

The subdomain keeps track of the data set (the `.arc` file) and of the output
directories:

- \arcane{ISubDomain::caseName()}, \arcane{ISubDomain::caseFullFileName()},
  \arcane{ISubDomain::fillCaseBytes()}, \arcane{ISubDomain::setCaseName()}:
  information on the data set. \arcane{ISubDomain::setCaseName()} must be
  called before initialization.
- \arcane{ISubDomain::caseDocument()}: the XML document of the data set.
- \arcane{ISubDomain::caseOptionsMain()}: the global options of the data set
  (\arcane{CaseOptionsMain}).
- \arcane{ISubDomain::commonVariables()}: the "standard" variables (time, time
  step, ...).

Three output directories can be set (the directories must exist and the methods
must be called before initialization):

- \arcane{ISubDomain::exportDirectory()} / \arcane{ISubDomain::setExportDirectory()}:
  the base directory for exports (protections and restarts).
- \arcane{ISubDomain::storageDirectory()} / \arcane{ISubDomain::setStorageDirectory()}:
  the base directory for exports requiring archiving. If not set,
  \arcane{ISubDomain::exportDirectory()} is used.
- \arcane{ISubDomain::listingDirectory()} / \arcane{ISubDomain::setListingDirectory()}:
  the base directory for listings (logs, execution information).

## Creation and Lifecycle {#arcanedoc_core_types_subdomain_lifecycle}

A subdomain is created from a \arcane{SubDomainBuildInfo} object, which
provides:
- the parallel manager of the subdomain
  (\arcane{SubDomainBuildInfo::parallelMng()}),
- the subdomain index in the session (\arcane{SubDomainBuildInfo::index()}),
- the data set: file name (informative) and content
  (\arcane{SubDomainBuildInfo::caseBytes()}),
- optionally, the parallel manager of all the replicas
  (\arcane{SubDomainBuildInfo::allReplicaParallelMng()}) for domain
  replication.

The creation goes through \arcane{ISession::createSubDomain()}:

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
| Mesh setup | \arcane{ISubDomain::allocateMeshes()}, \arcane{ISubDomain::readOrReloadMeshes()}, \arcane{ISubDomain::initializeMeshVariablesFromCaseFile()}, \arcane{ISubDomain::doInitMeshPartition()} | Allocation and reading of the meshes, initialization of the mesh variables from the data set, application of the initial partition. |
| Restart support | \arcane{ISubDomain::setIsContinue()} / \arcane{ISubDomain::isContinue()} | Flag indicating that a restart is being performed. \arcane{ISubDomain::setIsContinue()} must be called before \arcane{ISubDomain::allocateMeshes()} so that the meshes are loaded from a protection instead of the data set. |
| Run | (application code) | The application entry points are executed through the time loop. |
| Destroy | \arcane{ISubDomain::doExitModules()}, \arcane{ISubDomain::destroy()} | Execution of the *exit* entry points, then destruction of all the subdomain objects (the variable manager is destroyed last, since every object may contain variables). |

\arcane{ISubDomain::isInitialized()} / \arcane{ISubDomain::setIsInitialized()}
indicate whether the subdomain has been initialized.
\arcane{ISubDomain::onDestroyObservable()} allows registering a notification
that is called before the subdomain is destroyed.

## Getting the SubDomain from the Code {#arcanedoc_core_types_subdomain_get}

Nearly every object owned by the subdomain provides a `subDomain()` accessor to
climb back to it:

- services: `subDomain()` inherited from the service base class,
- modules: \arcane{IModule::subDomain()} inherited from \arcane{IModule},
- entry points: \arcane{IEntryPoint::subDomain()} inherited from \arcane{IEntryPoint},
- meshes: \arcane{IMesh::subDomain()},
- variables: \arcane{IVariable::subDomain()},
- item families: \arcane{IItemFamily::subDomain()},
- case options: \arcane{ICaseOptions::subDomain()}.

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
- It is *not* the application: \arcane{ISubDomain::application()} gives the
  process-wide objects. The subdomain is where the identity of the computation
  unit lives (\arcane{ISubDomain::subDomainId()} is the rank).
- The ordering of the initialization methods matters:
  \arcane{ISubDomain::setCaseName()}, \arcane{ISubDomain::setExportDirectory()},
  \arcane{ISubDomain::setInitialPartitioner()} before initialization;
  \arcane{ISubDomain::setIsContinue()} before \arcane{ISubDomain::allocateMeshes()}.
- The subdomain is the seam for the advanced execution modes: domain
  replication (several subdomains per session), restarts (protections), and
  direct execution (tests), which all plug in through its setters
  (\arcane{ISubDomain::directExecution()} / \arcane{ISubDomain::setDirectExecution()}).

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_array_views
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_mesh
</span>
</div>
