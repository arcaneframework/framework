# Changelog %Arcane v2.X.X {#arcanedoc_news_changelog20}

[TOC]

This page contains the new features of each version of %Arcane v2.X.X.



___
## Arcane Version 2.22.0 (03 March 2021) {#arcanedoc_news_changelog_version2220}

### New Features/Improvements:

- Support for protections/recoveries for AMR meshes
- Internal modifications for accelerator support, notably support for graphics
  cards (GPGPU).
- Removal of entity classes and families for Arcane::DualNode and Arcane::Link.
  These classes can be replaced by using Arcane::DoF.
- Partial support for mesh files in 'msh' format version 4.1.

### Changes:

- [IMPORTANT] This version uses the new connectivity structures by default. This
  should normally not impact existing codes provided they do not use the
  internal %Arcane classes that manage entities (such as Arcane::ItemInternal).
- Execution statistics on time spent in different entry points and modules
  displayed at the end of the calculation now contain the cumulative total over
  all executions (in case of using protection/recovery) and not just the
  statistics for the current execution.
- Simplification of the various mechanisms for initializing and executing code
  using %Arcane (see \ref arcanedoc_execution_launcher)

### Fixes:

- Fixes a potential parallel issue during the creation of the directory for
  protections. It was possible for some processes to attempt to access this
  directory before its creation.
- Fixes the usage of the `arcane_convert_mesh` script used for mesh conversion.
- Re-enables the generation of documentation for services and modules provided
  by %Arcane

___
## Arcane Version 2.20.0 (07 January 2021) {#arcanedoc_news_changelog_version2200}

### New Features/Improvements:

- Addition of a new interface 'Arcane::IAsyncParticleExchanger' allowing
  asynchronous particle exchange. A new service `AsyncParticleExchanger`
  implements this interface.
- Adds the possibility to delete additional memory possibly used by a variable
  via the method Arcane::IVariable::shrinkMemory()
- Start of support for 2D Cartesian mesh refinement. It is now possible to call
  Arcane::ICartesianMesh::refinePatch2D() to refine a part of the mesh
- Support for Arccore::MessagePassing::MessageId in serialization messages
  (Arccore::MessagePassing::ISerializeMessage)
- Support for Papi version 6.0.0 and recent PETSc versions for `aleph`

### Changes:

- Unification of documentation between the CEA part and the common part
- Support for parallel unit tests (see
  \ref arcanedoc_debug_perf_unit_tests_parallel)
- Modifications to the classes Arcane::IData, Arcane::ISerializedData to use
  Arccore::Ref for managing references
- Start of redesign of variable creation and memory management to eventually
  allow developers to add their own variables.

### Fixes:

- Fixes crash when using PTScotch during partitioning in case of single-node
  redistribution.
- Correctly names the 'ArcaneConfigVersion.cmake' file to be able to use a
  version number for %Arcane in the CMake command `find_package()`.

___
## Arcane Version 2.19.0 (30 June 2020) {#arcanedoc_news_changelog_version2190}

\note Following changes in serialization, it is generally not possible to resume
a calculation started with an earlier version using this version of %Arcane.

### Changes:

- To prepare for the modularization of %Arcane, retrieving the subdomain
  (Arcane::ISubDomain) is deprecated via many classes (Arcane::IMesh,
  Arcane::IVariable, Arcane::ICaseOptions, ...). Codes that require an instance
  of Arcane::ISubDomain must now explicitly pass it to the methods or classes
  using it. Services and modules continue to have access to subdomains.
- Use of Arcane::MeshHandle instead of Arcane::IMesh. With the possibility of
  using services to create meshes, meshes are no longer created before services.
  The Arcane::MeshHandle class allows managing a reference to a mesh before it
  exists. This class must notably be used in the constructors of modules and
  services. The method Arcane::MeshHandle::mesh() allows retrieving the
  associated mesh if it has already been created.
- Use of the Arccore::Ref class to manage references to Arcane::IParallelMng.
  Consequently, the method Arcane::IParallelMng::createSubParallelMng() is
  declared deprecated and Arcane::IParallelMng::createSubParallelMngRef() must
  now be used.
- Modification of Arccore::Span2::operator[]() to return an Arccore::Span
  instead of Arccore::ArrayView. This allows consistency with other methods of
  Arccore::Span2.
- Modification of Arccore::ISerializer to support direct writing of arrays and
  check consistency between reading and writing a given value. This requires
  that the calls `reserve()/put()/get()` are consistent. For example, if the
  method Arccore::ISerializer::reserveArray() is used,
  Arccore::ISerializer::putArray() and Arccore::ISerializer::getArray() must
  subsequently be used. Following this modification, the methods
  Arccore::ISerializer::get() and Arccore::ISerializer::put() which took an
  Arccore::Span or Arccore::ArrayView as argument are deprecated.
- Use of a specific macro ARCANE_CHECK_MATH to test the validity of Arcane::math
  mathematical operations. Previously, this was done via the ARCANE_CHECK macro,
  but this could introduce side effects when comparing two calculations with and
  without this verification mode.

### New Features/Improvements:

- Numerous improvements to message exchange:
  - Support for all Arcane::IParallelMng methods in hybrid and shared memory
    modes. In particular, some serialization methods were missing.
  - Support for user tags in point-to-point methods (send/receive)
  - Use of specific types to manage tags (Arccore::MessageTag) and ranks
    (Arccore::MessageRank). These types replace the 'Int32' type previously
    used.
  - Unification of point-to-point method management. The necessary parameters
    are managed by the Arccore::PointToPointMessageInfo class.
  - Support for MPI methods using an `MPI_Message` (`MPI_Mprobe`, `MPI_Improbe`,
    `MPI_Imrecv`, `MPI_Mrecv`). These methods ensure that an `MPI_Recv`
    corresponds to the `MPI_Probe` used previously. These mechanisms are also
    accessible in hybrid mode.
- Support for mesh partitioning in shared memory and hybrid modes
- Adds to the `Metis` partitioner a mode allowing a single process to be used
  for partitioning. Previously, a minimum of two processes had to be used (due
  to a bug in `ParMetis`)
- Correction of the algorithm ensuring that there are no empty partitions after
  using `ParMetis`.

- Preparation of Arccore::ISerializer to eventually support other data types
  (for example, 'Float32' or 'Float16')
- Records a hash of the variable name during serializations. This ensures that
  the correct variable is being deserialized.
- Various improvements to the JSON reader/writer
- Generates information in the logs after a repartitioning about the subdomains
  connected to a subdomain.
- Support for compilation via the 'spack' tool.
- In the dataset, adds the possibility to put a block `<comment>` below the
  root. This is useful if a module is not active but you want to keep its
  options. You simply need to place this option block between the `<comment>`and
  `</comment>` tags.
- [.Net] Generates `nuget` packages for wrapping usage. It is therefore possible
  to directly reference these packages rather than going to look for the `.dll`
  files directly. The use of `nuget` packages also allows automatic dependency
  management.
- [.Net] Transition to .Net Core version 3.1.

The following points are under development but not finalized:

- Management of mesh creation as a service. Eventually, only this mode will be
  available. It will notably allow the creation of specific mesh structures (for
  example, Cartesian).
- Redesign of initialization to allow %Arcane to be used without going through a
  loop in time

___
## Arcane Version 2.18.0 (09 December 2019) {#arcanedoc_news_changelog_version2180}

This version includes the following developments:

- Improvement of the implementation of the Arcane::HPReal class, which fixes
  some edge cases and adds support for multiplication and division
- Creation of a universal Arccore::Ref class to manage the lifetime of objects
  such as services. This class automatically destroys an object when there are
  no more references to it. Dynamically allocated services now use this class.
  For example:
  ```cpp
  using namespace Arcane;
  ISubDomain* sd = ...;
  ServiceBuilder<IDataReader> sb(sd);
  IDataReader* old_reader = sb.createInstance(...); // Deprecated
  Arccore::Ref<IDataReader> new_reader = sb.createReference(...); // New mechanism.
  ```
  Via this Arccore::Ref class, an object no longer needs to be explicitly
  destroyed. The method Arccore::makeRef() allows creating a reference from a
  pointer allocated by 'operator new'.
- Partial redesign of the dataset option reading mechanism. The ultimate goal is
  to be able to use something other than XML for data input. The modifications
  mainly concern the base classes managing options and should not affect user
  code.
- Detects dataset options under the root that are not read and returns an error.
  The environment variable **ARCANE_ALLOW_UNKNOWN_ROOT_ELEMENT** allows
  displaying a warning instead of an error if set.
- [.Net] Transition to .Net Core version 3.0.
- [.Net] C# support for classes managing materials and environment and scalar
  variables on constituents.
- [.Net] Support for launching C# user extensions via the `dotnet` tool instead
  of `mono`, which allows debugging C# code via, for example, *Visual Studio
  Code*.
- Transition to Lima version 6.38.0, which allows reading files in `mli2`
  format. This format uses a recent version of *HDF5* and optimizes, in
  particular, the size of mesh files when there is a large number of empty
  groups.




___
## Arcane Version 2.17.0 (09 October 2019) {#arcanedoc_news_changelog_version2170}

This version includes the following developments:
- [.Net] Redesign of the .Net environment to allow compilation either with
  `mono` or with the Microsoft `dotnet` implementation.
- [.Net] Standardization of initialization with or without the use of the .Net
  runtime.
- Addition in the 'samples' directory of an 'EOS' example showing how to make
  C++ services or classes accessible in C#.
- Strict application of XML conversion rules for reading the `active` attribute
  in the `<module>` element indicating whether a module is active or not. Now,
  only the values `true`, `false`, `0` or `1` are allowed. Previously, any value
  other than `false` was considered `true`.
- Support in MPI profiling via the OTF2 library for cases with load balancing
  and back-propagation.
- Possibility to destroy additional meshes via the Arcane::IMeshMng class:
  ```cpp
  Arcane::ISubDomain* sd = ...;
  Arcane::IMesh* mesh_to_destroy = ...;
  sd->meshMng()->destroyMesh(mesh_to_destroy);
  ```
  The pointer `mesh_to_destroy` must no longer be used after calling the
  destruction method.
- Support in Arcane::mesh::BasicParticleExchanger for particles that are not in
  meshes (those for which Arcane::Particle::cell() returns a null mesh)
- Adds options in the 'BasicParticleExchanger' service to allow choosing the
  maximum number of messages to perform before reductions
- Adds an environment variable `ARCANE_VARIABLE_SHRINK_MEMORY` for testing,
  which, if set to `1`, resizes the memory allocated by variables as little as
  possible after changing the number of entities
- In `Aleph`, MPI communicators are destroyed in the destructor of
  `AlephKernel`. This had been disabled for reasons of compatibility with PETSc
  and old MPI versions.




___
## Arcane Version 2.16.0 (18 July 2019) {#arcanedoc_news_changelog_version2160}

This version includes the following developments:

- Addition of internal profiling for entry points and MPI calls. The trace
  format depends on the value of the environment variable that activates this
  feature: **ARCANE_MESSAGE_PASSING_PROFILING=OTF2** or **JSON** (see
  environment variable documentation and performance analysis).
- Redesign of the C# wrapper. The wrapper is now composed of several modules
  (Arcane.Core, Arcane.Hdf5, and Arcane.Services) and the wrapped methods now
  use C# coding rules: they start with a capital letter.
- Addition of an Arcane::MeshHandle class to manage meshes before their
  effective creation. Eventually, this will allow meshes to be deleted and
  created via services.




___
## Arcane Version 2.15.0 (13 June 2019) {#arcanedoc_news_changelog_version2150}

This version includes the following developments:

- Support for MPI messages larger than 2GB. This only concerns MPI_Send/MPI_Recv
  type messages and only if passing through Arcane::ISerializer.
- Completes the implementation of Arcane::IParallelMng in shared memory mode.
  All methods except Arcane::IParallelMng::createSubParallelMng() are
  implemented. The classes managing this parallelism mode are renamed and start
  with *SharedMemory* instead of *Thread*.
- Renaming of classes managing message exchange in MPI+Shared Memory mode. These
  classes have a name starting with *Hybrid* instead of *MpiThread*.
- Support for .NetCore in addition to `mono` for managing *AXL* files. Projects
  managing *AXL* files can therefore use either .NetFramework 4.5 (with mono) or
  .NetCoreApp 2.2 (with .NetCore). It is now essential to have the `msbuild`
  tool to compile with `mono`. By default, the .NetCore implementation is used.
- The Arccore::String class now considers arguments of type `const char*` to be
  encoded in UTF-8 and no longer in ISO-8859-1.
- The Arccore::String class now internally manages character strings larger than
  2GB. Consequently, the method Arccore::String::len() is deprecated and must be
  replaced by the method Arccore::String::length() which returns an *Int64*.
- It is now possible to construct instances of Arccore::String from the C++17
  std::string_view class. Consequently, the methods of Arccore::String that took
  a `const char*` and a length as arguments are deprecated and replaced by
  methods taking std::string_view as arguments.
- A reproducible mode for ParMetis has been developed that ensures the same
  partitioning between two executions by grouping the graph on a single
  processor. This mode should not be used when the number of cells is greater
  than a few tens of millions.

___
## Arcane Version 2.14.0 (04 March 2019) {#arcanedoc_news_changelog_version2140}

This version includes the following developments:

- Integration of IFPEN developments concerning degree of freedom type entities (
  Arcane::DoF class).
- Possibility to redirect listing outputs both to standard output (`stdout`) and
  to a file with the option of not having different verbosity levels. Following
  this change, if the environment variable `ARCANE_PARALLEL_OUTPUT` is set,
  subdomain 0 writes the listing to a file like the other subdomains (the files
  are named `output...`). To specify that you want to write a listing file, you
  must set the environment variable `ARCANE_MASTER_HAS_OUTPUT_FILE` or call the
  method Arcane::ITraceMngPolicy::setIsMasterHasOutputFile() before
  initialization (for example, by overriding Arcane::MainFactory).
- Transition to UTF-8 for all sources (%Arcane, %Arccore,...)
- Moving the 'arccore' sources to its own GIT repository and moving the sources
  of `arcane/dof` to `arcane/mesh`

___
## Arcane Version 2.13.0 (21 January 2019) {#arcanedoc_news_changelog_version2130}

This version includes the following developments:

- **[INCOMPATIBILITY]** Modification of the *begin()* and *end()* methods for
  array and array view classes (Arccore::ArrayView, Arccore::ConstArrayView,
  Arccore::Span, Arccore::Array) to return an iterator instead of a pointer. If
  you want to retrieve a pointer, you must use the *data()* method instead.
- **[INCOMPATIBILITY]** Creation of Arccore::Array instances is forbidden. You
  must use either Arccore::UniqueArray or Arccore::SharedArray. The
  Arccore::Array class must be used by reference as an argument or return value.
- Partial transition of %Arcane sources to UTF-8
- Transition to HDF5 version 1.10+ (instead of 1.8)
- Support for MED format meshes (format used by CEA/DEN, notably by the Salomé
  platform)
- Modification of the 'axldoc' generator to generate Doxygen files in Markdown
  format with the '.md' extension instead of .dox files




___
## Arcane Version 2.12.0 (11 December 2018) {#arcanedoc_news_changelog_version2120}

This version includes the following developments:

- Improvement of variable visualization with the latest versions of totalview
  and addition of visualization for scalar variables
- Creation of an Arccore::Span2 class to manage views on 2D arrays with 64-bit
  sizes.
- Internal redesign of protections/recoveries to allow changing the number of
  subdomains in recovery.




___
## Arcane Version 2.11.0 (18 October 2018) {#arcanedoc_news_changelog_version2110}

This version includes the following developments:

- Support for arrays (Arccore::Array) and character strings (Arccore::String)
  exceeding 2GB. Internally, the number of elements is now stored in 64 bits
  instead of 32 bits.
- Addition of the 'LargeArrayView' and 'ConstLargeArrayView' classes which are
  identical to 'ArrayView' and 'ConstArrayView' but use a 64-bit size.




___
## Arcane Version 2.10.1 (04 October 2018) {#arcanedoc_news_changelog_version2101}

This version includes the following developments:

- Possibility to specify multiple interfaces for services via the
  ARCANE_REGISTER_SERVICE() macro. Previously, this was only possible via 'axl'
  files.
- Singleton services implementing multiple interfaces are created only once, and
  the interfaces therefore refer to the same instance. Previously, there were as
  many instances as there were interfaces.
- It is possible to specify in the 'axl' file that a service is a singleton (see
  \ref arcanedoc_core_types_service_desc).
- Possibility to specify default values by category via a new **defaultvalue**
  element in 'axl' files. The category used during execution can be set for each
  code via the method Arcane::ICaseDocument::setDefaultCategory(). The page
  \ref arcanedoc_core_types_axl_caseoptions_struct indicates how to add these
  default values.
- Possibility to load singleton services in the dataset like modules via a new
  **services** element in the `<arcane>`element (see
  \ref arcanedoc_core_types_casefile_arcaneelement).




___
## Arcane Version 2.10.0 (September 2018) {#arcanedoc_news_changelog_version2100}

This version includes the following developments:

- Creation of a new component '%Arccore' grouping the common %Arcane part with
  Alien. This component contains part of 'arcane_utils' and 'arcane_mpi'.
  Consequently, base classes such as Arccore::String, Arccore::ArrayView,
  Arccore::Array are now in the 'Arccore' namespace instead of 'Arcane'. Via the
  C++ 'using' mechanism, these classes are also available in the Arcane
  namespace, so there should be no incompatibility with existing source code.
  The only condition is not to explicitly declare %Arcane types but to use the
  header file 'arcane/utils/UtilsTypes.h'.
- Use of the 'libxml2' library to manage XML instead of the 'XercesC' library.
  This modification was made for two features not available in 'XercesC':
  XInclude support with XmlSchema and UTF-8 encoding of character strings.
- Construction of 'Arccore::Array' is deprecated. You must explicitly use
  either 'Arccore::UniqueArray' or 'Arccore::SharedArray'.
- Separation of the implementation of Arccore::UniqueArray from that of
  Arccore::SharedArray to avoid retaining information that is only useful for
  one or the other of these classes.
- Possibility to specify a number of threads between 1 and the maximum number of
  threads allocated during an Arcane::Parallel::ForEach loop. This is done via
  the Arcane::ParallelLoopOptions class.
- The 'arcane.pc' file generated for pkg-config is deprecated and no longer
  contains all the libraries used by %Arcane. To have an equivalent feature, you
  must use 'cmake' with the '--find-package' option.
- Redesign of iterator management on array classes (Arccore::Array,
  Arccore::ArrayView, Arccore::ConstArrayView). The current iterators that
  returned a pointer are declared deprecated and replaced by an object of type
  Arccore::ArrayIterator. Returning a pointer could pose a problem in case of
  inheritance. For example:
```cpp
class A { ... };
class B : public A { ... };

Arcane::Array<B*> array_of_b;
for( A* a : array_of_b){
  // Plantage si sizeof(A)!=sizeof(B)
}
```
  The new iterator is shared among the three array classes and is of type
  std::random_iterator.
  Depending on the use case, the current code may be changed as follows:
  
  - Using begin() to retrieve a pointer to the start of the array: use data()
    instead:
```cpp
Arcane::ArrayView<Int32> a;
Int32* v = a.begin(); // Obsolète
Int32* v = a.data();  // OK.
```

  - Using begin()/end() in STL algorithms. In this case, you must replace these
    methods with std::begin() and std::end():
```cpp
Arcane::ArrayView<Int32> a;
std::sort(a.begin(),a.end()); // Obsolète
std::sort(std::begin(a),std::end(a)); // OK
```

  - Using in the case of a C++11 for-range loop. In this case, you must use the
    array's range method. For example:
```cpp
Arcane::ArrayView<Int32> a;
for( Int32 x : a) {} // Obsolète
for( Int32 x : a.range()) {} // OK
```




___
## Arcane Version 2.9.1 (June 2018) {#arcanedoc_news_changelog_version291}

This version includes the following developments:
- Possibility to change the default values of the dataset options. For more
  information, refer to section \ref
  arcanedoc_core_types_axl_caseoptions_default_values.
- Adds the valueIfPresentOrArgument() method for simple, enumerated, or extended
  options. This allows replacing the following code:
```cpp
Real x = 3.2;
if (options()->myOption.isPresent())
  x = options()->myOptions();
```
with the following code:
```cpp
Real x = options()->myOption.valueIfPresentOrArgument(3.2);
```
- 64-bit support for the non-blocking particle exchanger (service *
  *NonBlockingParticleExchanger** implementing Arcane::IParticleExchanger). This
  allows exceeding 2^31 particles during an exchange.
- Adds the Arcane::ITimeLoopMng::stopReason() method, allowing knowledge of the
  reason for the code's termination. In particular, it is now possible to know
  if the last iteration is being executed when a maximum number of iterations is
  specified.
- In the dataset, in the list of modules to activate, it is now possible to
  specify the name (translated or name) of the Module's XML element instead of
  the module name. For example, for the %Arcane protection/recovery module,
  whose name is 'ArcaneCheckpoint', we currently have:
```xml
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <modules>
      <module name="ArcaneCheckpoint" actif="true"/>
    </modules>
  </arcane>
  ...
  <arcane-protections-reprises>
    <periode>3</periode>
  </arcane-protections-reprises>
</cas>
```
and now it is possible to replace it with:
```xml
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <modules>
      <module name="arcane-protections-reprises" actif="true"/>
    </modules>
  </arcane>
  ...
  <arcane-protections-reprises>
    <periode>3</periode>
  </arcane-protections-reprises>
</cas>
```




___
## Arcane Version 2.8.0 (January 31, 2018) {#arcanedoc_news_changelog_version280}

This version includes the following developments:

- Addition for unit tests of two macros #ASSERT_NEARLY_ZERO and
  #ASSERT_NEARLY_ZERO_EPSILON, which allow comparing a value to zero.
- Compile-time determination of the types 'Arcane::Int16', 'Arcane::Int32',
  and 'Arcane::Int64'. Because of this, the file 'ArcaneGlobal.h' no longer
  includes the file 'limits.h'.
- By default, the length units specified in Lima format files (.unf, .mli, ...)
  are now always taken into account. Previously, it was necessary to set the
  value 1 to the 'use-unit' attribute (in English) or 'utilise-unite' (in
  French) in the <mesh> (in English) or <maillage> (in French) element. To
  recover the previous behavior, you must set the value '0' to this attribute.
  For example in French:

  ```xml
  <maillage utilise-unite="0">
  ...
  </maillage>
  ```

  or in English:

  ```xml
  <mesh use-unit="0">
  ...
  </mesh>
  ```




___
## Arcane Version 2.7.0 (October 20, 2017) {#arcanedoc_news_changelog_version270}

This version includes the following developments regarding materials and
environment:
- possibility to iterate only over pure or impure cells of a material or an
  environment.
- new mechanisms for looping over materials and environments
- start of vectorization support for materials and environment. For now, this
  support only concerns scalar variables and only on environment.

For more information on these concepts, refer to page
\ref arcanedoc_materials_loop.




___
## Arcane Version 2.6.1 (September 18, 2017) {#arcanedoc_news_changelog_version261}

This version includes the following visible developments:
- In the dataset options, it no longer tolerates invalid characters at the end
  of the option string. For example, the string '12a3' was considered valid for
  an integer type option and its value was '12'. Now an error is returned in
  this case.

This version includes internal developments aimed at eventually removing the old
connectivity access mechanisms.




___
## Arcane Version 2.6.0 (August 22, 2017) {#arcanedoc_news_changelog_version260}

This version activates the new connectivity access mechanism. The old mechanism
remains accessible. Since the two mechanisms use different memory management,
connectivities are allocated using both the new and the old mechanism, resulting
in an increase of about 1KB of memory per cell.
For more information, refer to page
\ref arcanedoc_entities_connectivity_internal.




___
## Arcane Version 2.5.2 (July 25, 2017) {#arcanedoc_news_changelog_version252}

This version contains the following developments:

- addition of an event management mechanism (see file "arcane/utils/Event.h" and
  the EventObservable class for a usage example)
- addition of events during variable addition/removal
  (Arcane::IVariableMng::onVariableAdded() and
  Arcane::IVariableMng::onVariableRemoved()) and during
  synchronizations (Arcane::IVariableSynchronizer::onSynchronized()).
- possibility to specify a static partitioner for loops using multi-threading.
  This allows for deterministic (i.e., repeatable) behavior between multiple
  executions provided the same number of threads is used.
```cpp
#include <arcane/Concurrency.h>
using namespace Arcane;
ParallelLoopOptions opt;
opt.setPartitioner(ParallelLoopOptions::Partitioner::Static);
TaskFactory::setDefaultParallelLoopOptions(opt);
```

- addition of a named barrier mechanism to ensure that all processes reach the
  same barrier:
```cpp
#include <arcane/Parallel.h>
using namespace Arcane;
IParallelMng* pm = ...;
MessagePassing::namedBarrier(pm,"MyBarrier");
```

- possibility to mix declarations of material and environment variables (for scalar
  variables only). In this case, all variables are allocated on materials and
  environment. By default, this is not active because it consumes memory
  unnecessarily.
```cpp
using namespace Arcane::Materials;
IMeshMaterialMng* mm = ...;
mm->setAllocateScalarEnvironmentVariableAsMaterial(true);
MaterialVariableCellReal mat_var(MaterialVariableBuildInfo(mm,"Var1"));
EnvironmentVariableCellReal env_var(MaterialVariableBuildInfo(mm,"Var1"));
// 'mat_var' and 'env var' are the same variables.
```





___
## Arcane Version 2.5.1 (May 04, 2017) {#arcanedoc_news_changelog_version251}

This version contains only internal developments and does not add features
directly accessible to users.




___
## Arcane Version 2.5.0 (March 15, 2017) {#arcanedoc_news_changelog_version250}

This version integrates developments to decouple connectivity management from
entity management. For memory saving reasons, the current mechanism allows all
connectivities to be available for each entity type. Potentially, this means,
for example, that a particle and a cell can have the same connectivities. For
more information on the new mechanisms, refer to the following page
\ref arcanedoc_entities_connectivity_internal. In this 2.5.0 version, the new
connectivities are not active (which corresponds to the '
--with-legacy-connectivity' configuration mode described on the previous link
page), so the behavior of codes using %Arcane should not change.

Other new features are:
- addition of an Item::isShared() method to indicate if an entity is shared by
  multiple subdomains. For example:

```cpp
Cell cell = ...;
if (cell.isShared())
  info() << "Cell is shared";
```

- synchronization of multiple variables in a single MPI message is now active by
  default. For example:

```cpp
VariableCellReal temperature = ...;
VariableCellReal3 cell_center = ...;
VariableCollection vars;
vars.add(temperature);
vars.add(cell_center);
mesh()->cellFamily()->synchronize(vars);
```
Note that if the collection specified in the parameter contains partial
variables, the synchronization is done as before, variable by variable.




___
## Arcane Version 2.4.2 (January 13, 2017) {#arcanedoc_news_changelog_version242}

- Adds the possibility to configure traces during execution by specifying a
  character string in the trace configuration format (as in the documentation
  \ref arcanedoc_execution_traces). For example:
```cpp
using namespace Arcane;
String x = "<?xml version=\"1.0\"?>\n"
 "<arcane-config>\n"
 "<traces>\n"
 "<trace-class name='MyTest2' info='true' debug='med' print-elapsed-time='true' print-class-name='false'/>\n"
 "</traces>\n"
 "</arcane-config>\n";
ISubDomain* sd = ...;
ITraceMng* tm = sd->traceMng();
sd->application()->getTraceMngPolicy()->setClassConfigFromXmlBuffer(tm,x.utf8());
```

- Adds the possibility to display or not display the message class name in the
  traces for each trace message, as well as the elapsed time. For example:
```cpp
using namespace Arcane;
ITraceMng* tm = ...;
TraceClassConfig tcc = tm->classConfig("MyTest");
tcc.setFlags(Trace::PF_ElapsedTime|Trace::PF_NoClassName);
tm->setClassConfig("MyTest",tcc);
```
- Adds an Arcane::IMeshUtilities::mergeNodes() method allowing the merging of
  nodes two by two.
- The internal entity compaction mechanism has been rewritten to be more easily
  configurable by entity family and more easily extensible if a new entity
  family type is added.




___
## Arcane Version 2.4.1 (December 01, 2016) {#arcanedoc_news_changelog_version241}

- Support for the Arcane::MeshMaterialVariableRef::synchronize() method for
  variables only on environment.
- Adds a method on material variables to fill partial values with the value of
  the parent component cell. This allows filling material values with environment
  values or environment values with global values. The method is called
  Arcane::Materials::MeshMaterialVariableRef::fillPartialValuesWithSuperValues().
- Adds constant STL iterators for the Arcane::ItemVectorView class. This allows
  using STL algorithms with this class.
  For example:
```cpp
using namespace Arcane;
ItemVector v1;
ItemGroup group1;
ItemVectorView group_view = group1.view();
for( Item item : v1.view() ){
  // Regarde si 'item' est dans le groupe 'group1'.
  auto iter = std::find(group_view.begin(),group_view().end,item);
  if (iter!=group_view.end()){
    // Trouvé.
  }
}
for( Item item : group1.view() ){
  info() << "Item=" << ItemPrinter(item);
}
```
- Adds the possibility to specify its own connectivity calculation function for
  Arcane::ItemPairGroup. The ItemPairGroup class documentation contains an
  example of such a calculation.
- Adds several options in the partitioning services:
  - for Parmetis, the possibility to write the graph to a file and specify the
    tolerance,
  - for PTScotch, the possibility to write the graph to a file and check the
    graph consistency.




___
## Arcane Version 2.4.0 (November 2016) {#arcanedoc_news_changelog_version240}

- Support for variables only on environment. These variables are used like
  material variables but only have values on environment and global cells.
  Therefore, they should not be indexed with material cells (MatCell) under
  penalty of causing an illegal memory access. The declaration of these
  variables is done as follows:
```cpp
using namespace Arcane;
EnvironmentVariableCellReal pressure(VariableBuildInfo(mesh,"Pressure"));
IMeshEnvironment* env = ...;
ENUMERATE_ENVCELL(ienvcell,env){
  pressure[ienvcell] = 2.0;
}
```
- Support for repartitioning and load balancing while preserving material
  information. Partial values of material and environment cells are also
  preserved.
- Internal redesign of entity exchange management during repartitioning. The
  goal is to be able to more easily add new entity families or change the
  processing of one of them. This modification has no impact on existing code.




___
## Arcane Version 2.3.9 (September 2016) {#arcanedoc_news_changelog_version239}

- Improvement of vectorization {#arcanedoc_news_changelog_version239_simd}
- Possibility to choose a specific allocator (implementing the IMemoryAllocator
  interface) for the Array and UniqueArray classes. An implementation managing
  alignment (Arcane::AlignedMemoryAllocator) is available and guarantees that
  the address of the allocated memory is a multiple of a certain value. For now,
  the only valid value for alignment is 64, and you must use the
  Arcane::AlignedMemoryAllocator::Simd() allocator to retrieve this allocator
  with this alignment. In addition to aligning the memory, this allocator
  guarantees that the size of the allocated memory is a multiple of the
  alignment.
  For example, the following code guarantees alignment:
```cpp
using namespace Arcane;
UniqueArray<Real> x(AlignedMemoryAllocator::Simd());
x.resize(25);
// &x[0] is a multiple of 64.
// A real taking 8 bytes, the capacity of x is a
// multiple of (64 / 8), so x.capacity()>=32 (since 32 is the
// first multiple of 8 greater than 25).
```
  The 64-byte alignment allows using all vectorization mechanisms available in
  %Arcane to date, namely SSE, AVX, and AVX512.
- 1D and 2D array variables, as well as scalar and array variables on mesh
  entities, are now always allocated with the
  Arcane::AlignedMemoryAllocator::Simd() allocator.
- Entity indices of an ItemGroup are now always allocated with the
  Arcane::AlignedMemoryAllocator::Simd() allocator. Note that this is not the
  case for Arcane::ItemVector.
- The various macros used to manage enumeration with vectorization
  (#ENUMERATE_SIMD_CELL, #ENUMERATE_SIMD_NODE, ...) now require aligned arrays.
  If this is not the case, it causes an exception during execution.
```cpp
using namespace Arcane;
CellGroup cells = ...;
CellVector vec_cells = ...;
ENUMERATE_SIMD_CELL(ivcell,cells){ // OK because a CellGroup is always aligned
}
ENUMERATE_SIMD_CELL(ivcell,vec_cells){ // ERROR because a CellVector is not always aligned
}
```
- Using the Parallel::Foreach() methods on entity groups guarantees that
  iterations will be performed on multiples of 8 values. It is thus valid to
  write the following code:
```cpp
using namespace Arcane;
CellGroup cells = ...;
Parallel::Foreach(cells,[this](CellVectorView cvv){
  ENUMERATE_SIMD_CELL(ivcell,cvv){
    SimdCell cell = *ivcell;
    ...
  }
}
```
- Adds support for AVX512 type vectorization and removes vectorization specific
  to Intel Knight Corner (KNC) processors. AVX512 is supported starting from
  Xeon Skylake and on XeonPhi processors starting from Intel Knight Landing
  (KNL).
- Allows multiple types of vectorization at once via %Arcane's vectorization
  classes. For example, on Haswell machines that support AVX and SSE, the
  SSESimdReal and AVXSimdReal classes are always available. In previous versions
  of %Arcane, only the most performant vectorization type was available
  (AVX512 > AVX > SSE).

Various improvements {#arcanedoc_news_changelog_version239_misc}

- Support for the C++11 std::move() semantics for the UniqueArray class. This
  allows, among other things, returning UniqueArrays without performing memory
  copying, and implementing std::swap() in an optimized manner.
- Adds optimized versions to exchange the values of two mesh variables (scalar
  or array) via a swapValues() method. This method allows exchanging only the
  pointers containing the memory areas of the two variables without copying. For
  example
```cpp
using namespace Arcane;
VariableCellReal temperature = ...;
VariableCellReal old_temperature = ...;
  VariableCellArrayReal energy = ...; // 1D variable on meshes
VariableCellArrayReal old_energy = ...;
old_temperature.swapValues(temperature);
old_energy.swapValues(energy);
```
- Upgrade to version 1.11.4 of hwloc.
- [C#] In Arcane.Curves, support for curves with multiple values per iteration
  (2D curves).
- Uses real time (elapsed) instead of CPU time for end-of-calculation
  statistics. Real time is more accurate than CPU time and is independent of the
  number of threads used. It is possible to revert to using CPU time by setting
  the ARCANE_USE_VIRTUAL_TIMER environment variable.
