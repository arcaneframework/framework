# New Features {#arcanedoc_news_changelog}

[TOC]

This page contains the new features of each version of %Arcane v3.X.X.

The successive new features brought by versions of %Arcane prior to version 3
are listed here: \ref arcanedoc_news_changelog20

___

## Arcane Version 4.1+ (March 2026) {#arcanedoc_version410}

### New Features/Improvements

- Adds support for `backward-cpp` for displaying the stack in case of an error.
  This allows for line numbers and file names in addition to the function name
  (\pr{2248}, \pr{2333}, \pr{2484})
- Moves the accelerator API from %Arcane to %Arccore. All classes involved in
  the API are moved. The files in
  `arcane/accelerator/core` are moved to `arccore/common/accelerator` and those
  in `arcane/accelerator` are moved to
  `arccore/accelerator`. The header files in %Arcane are kept to ensure
  compatibility with existing code (\pr{2245}, \pr{2246}, \pr{2249}, \pr{2254},
  \pr{2256}, \pr{2257}, \pr{2258}, \pr{2259}, \pr{2263}, \pr{2263},\pr{2265},
  \pr{2266}, \pr{2269}, \pr{2270}, \pr{2271}, \pr{2272}, \pr{2273}, \pr{2274},
  \pr{2275}, \pr{2276}, \pr{2277}, \pr{2319}, \pr{2321}, \pr{2322}, \pr{2323},
  \pr{2324}, \pr{2325}, \pr{2326}, \pr{2328}, \pr{2329}, \pr{2330}, \pr{2332},
  \pr{2334}, \pr{2335}, \pr{2337}, \pr{2342}, \pr{2343}, \pr{2344}, \pr{2347},
  \pr{2356}, \pr{2358}, \pr{2359}, \pr{2360}, \pr{2361}, \pr{2368}, \pr{2371},
  \pr{2372}, \pr{2374}, \pr{2378}, \pr{2381}, \pr{2382}, \pr{2383}, \pr{2384},
  \pr{2385}, \pr{2386}, \pr{2392}, \pr{2396}, \pr{2401}, \pr{2402}, \pr{2425},
  \pr{2427}, \pr{2430}, \pr{2472})
- Adds support for `__float128` for Linux operating systems (\pr{2284},
  \pr{2399})
- Adds the possibility to create an intra-node communicator in
  \arcane{IParallelMng} (\pr{2290}, \pr{2292})
- Adds macros `ARCCORE_FATAL_IF()` and `ARCCORE_FATAL_IF()` to raise an
  exception if an assertion is not met (\pr{2303})
- Always creates boundary groups (`XMIN`, `XMAX`, ...) when using the Cartesian
  mesh generator (\pr{2312})
- Adds support for `Triangle10` type meshes (\pr{2318})
- Optimizes and improves AMR patch management for Cartesian meshes (\pr{2338},
  \pr{2366}, \pr{2370}, \pr{2440}, \pr{2456}, \pr{2459}, \pr{2476}, \pr{2494})
- Adds experimental geometric partitioner for initial partitioning (\pr{2340},
  \pr{2345}, \pr{2346}, \pr{2349}, \pr{2352}, \pr{2355})
- Improves the numbering of new meshes in the mesh subdivider. The new numbering
  respects the topological locality of the initial mesh (pr{2408})
- Optimizes Ensight outputs for polyhedral meshes (\pr{2415})
- Adds functions \arcane{math::normalizeL2()} for \arcane{Real2} and
  \arcane{Real3} (\pr{2455})
- Adds support for using MPI windows as allocators for variables (\pr{2482})
- Adds experimental support for changing the allocator of an \arcane{Array}
  (\pr{2493})

### Accelerator API

- Adds support for block-local memory (equivalent to the `__shared__` keyword in
  CUDA or HIP) (\pr{2281})
- Adds support for hierarchical parallelism (\pr{2287},\pr{2291}, \pr{2293},
  \pr{2295}, \pr{2296}, \pr{2297}, \pr{2298}, \pr{2299}, \pr{2313}, \pr{2316},
  \pr{2317}, \pr{2438}, \pr{2442}, \pr{2444}, \pr{2447}, \pr{2451}, \pr{2486})
- Adds support for cooperative kernel launching (via
  `cudaLaunchCooperativeKernel` or `hipLaunchCooperativeKernel`). This support
  is not available with the Sycl back-end (\pr{2282}, \pr{2439},
  \pr{2441},\pr{2452}, \pr{2461}, \pr{2469}, \pr{2475}, \pr{2479}, \pr{2481})
- Adds support for using a grid step for RUNCOMMAND_LOOP() and
  RUNCOMMAND_ENUMERATE() (\pr{2414}, \pr{2419})
- Simplifies and redesigns the internal implementation of accelerator
  mechanisms (\pr{2285}, \pr{2286}, \pr{2288}, \pr{2302})
- Adds support for memory pooling with HIP (\pr{2289},\pr{2307}, \pr{2310})
- Adds support for \arcaneacc{GenericSorter} for the Sycl back-end (\pr{2306})
- Automatically adds a barrier after \arcaneacc{GenericScanner} and
  \arcaneacc{GenericSorter} when using an asynchronous \arcaneacc{RunQueue}
  (\pr{2308})
- Enables memory pooling by default (\pr{2320})
- Adds public interface \arcane{IMemoryPool} to manage the memory pool
  (\pr{2376})
- Adds support for the `Int64` type in \arcane{ForLoopRange} loops (\pr{2364})
- Adds additional information in \arcaneacc{DeviceInfo} (\pr{2367}, \pr{2375},
  \pr{2377})
- Adds NCCL support for variable synchronizations (\pr{2373})
- Optimizes reduction management by removing the use of explicit memory copies.
  These copies are replaced by using pinned memory (\pr{2389}, \pr{2393},
  \pr{2492})
- With ROCM 7, uses a warp size fixed at compilation (\pr{2445}, \pr{2446})
- No longer uses `fastmod` for calculating indices in 2D and 3D loops. This
  allows tests to be removed and calculation to be accelerated (\pr{2450})
- Adds support for constructing \arcane{NumArray<T,MDDim1>} from an
  \arcane{Span} whose memory may be on CPU or GPU (\pr{2491})

### Changes

- Removes the template parameter `MinValue` in \arcane{SpanImpl}. This parameter
  was not used (\pr{2255})
- Makes certain methods of \arcane{Item} and derived classes `constexpr`
  (\pr{2261}, \pr{2264})
- Uses the STL thread implementation by default for
  \arcane{IThreadImplementation} instead of the TBB implementation \pr{2379}
- Removes support for compilation without enabling threads (\pr{2380})
- Uses `HIP` instead of `ROCM` for the CMake variable
  `ARCCORE_ACCELERATOR_MODE`. The old name is still available for
  compatibility (\pr{2457}).

### Corrections

- Correctly reorders edge nodes after modifying node `uniqueId()` (\pr{2262})
- Corrects the reading of the number of blocks at parallel nodes in the MSH
  reader (\pr{2278})
- Corrects compilation with CUDA versions 12.0 to 12.3 (\pr{2314})
- Corrects crash when synchronizing partial variables with an MPI "GPU-Aware"
  (\pr{2315})
- Prevents calling \arcane{IPrimaryMesh::allocateCells()} if
  \arcane{IMeshModifier::endUpdate()} has already been called (\pr{2354})

### Internal

- Removes historical implementation of \arcane{ItemInternalMap} (\pr{2247})
- Moves type-based subgroup management to an internal class (\pr{2353})
- Adds accelerator tests in %Arccore (\pr{2388}, \pr{2395}, \pr{2398},
  \pr{2404}, \pr{2411}, \pr{2412}, \pr{2462}, \pr{2480})
- Renames the namespace `Arcane::Accelerator::impl` to
  `Arcane::Accelerator::Impl` (\pr{2394}, \pr{2420})
- Various improvements in the `Neo` implementation (\pr{2464}, \pr{2465},
  \pr{2466}, \pr{2467}, \pr{2468}, \pr{2474}, \pr{2477})

### Compilation and Continuous Integration (CI)

- Removes support for old versions of TBB (before OneTBB 2021) : \pr{2267}
- Adds support for "self-hosted" runners (\pr{2279})
- Removes the CMake option `ARCANE_ADD_RPATH_TO_LIBS` (\pr{2304})
- Replaces the CMake option `ARCCORE_USE_MPI` with `ARCCORE_ENABLE_MPI`
  (\pr{2305})
- Replaces the CMake option `ARCCORE_WANT_TESTS` with `ARCCORE_ENABLE_TESTS`
  (\pr{2471})
- Adds CMake properties `INSTALL_RPATH_USE_LINK_PATH` and
  `BUILD_RPATH_USE_ORIGIN` for Arccore libraries (\pr{2331})
- Updates IFPEN workflows (\pr{2422}, \pr{2432}, \pr{2433})
- Updates the codecov workflow (\pr{2431})
- Prevents merging in a Pull Request without performing a "rebase" (\pr{2436},
  \pr{2437})
- Adds workflow for ROCM 7.2 (\pr{2448})
- Cleans up the compilation system for `alien/ArcaneInterface` (\pr{2487},
  \pr{2490})
- By default, disables the display of deprecated methods during compilation
  (\pr{2488})

### Arccore

- Adds new `common` component containing the collections and classes needed for
  the accelerator API (\pr{2268})
- In \arcane{ITraceMng}, uses `std::mutex` instead of \arcane{Mutex} (\pr{2253})
- Optimizes the construction/destruction of the \arcane{SmallArray} class
  (\pr{2242})
- Removes the retention of the number of elements of an \arcane{SpanImpl} when
  using a static dimension (\pr{2241})
- Moves JSON handling classes from `arcane_utils` to `arccore_common`
  (\pr{2391})

### Alien

- Updates default values with GMRES for PETSc (\pr{2251})
- Updates default values for BoomerAMG for Hypre (\pr{2365})
- Adds support for using CUDA in the Hypre and PETSc GAMG solver (\pr{2369},
  \pr{2403})
- Cleans up `PETScPrecomp.h` to accelerate compilation (\pr{2405})
- Improves GPU support for multiple back-ends (\pr{2409}, \pr{2410}, \pr{2413},
  \pr{2426}, \pr{2428}, \pr{2435}, \pr{2443}, \pr{2449}, \pr{2453}, \pr{2458},
  \pr{2473}, \pr{2485})

___

## Arcane Version 4.0.0 (October 15, 2025) {#arcanedoc_version4000}

Version 4.0.0 is identical to version 3.16.12 and only removes C++17 support.

___

## Arcane Version 3.16.12 (October 14, 2025) {#arcanedoc_version3160}

\note This version is the last to support the C++17 standard. Later versions
(4+) require C++20 support.

### New Features/Improvements

- Adds the possibility to modify dataset values from the command line
  (\pr{1988}, \pr{1994})
- Adds the \arcane{ItemLocalIdToItemConverter} class to obtain an \arcane{Item}
  from an \arcane{ItemLocalId}. This class is similar to
  \arcane{ItemInfoListView} but can be preserved throughout the calculation and
  remains valid after modification of the associated entity family (\pr{1971},
  \pr{2122})
- Adds \arcane{ItemTypeId} type constants for base entity types. These constants
  are prefixed by `ITI` (e.g., `ITI_Quad4` for the `IT_Quad4` type) (\pr{1972})
- Adds methods in \arcane{ParallelMngUtils} to create
  \arcane{Ref<ISerializeMessage>} instead of `ISerializeMessage*` (\pr{1999})
- Adds various improvements to the mesh subdivision service (\pr{2005})
- Adds support to be notified of entity additions or deletions (\pr{2013})
- Adds mesh writer in MSH 4.1 format (\pr{2020}, \pr{2026},\pr{2027}, \pr{2028},
  \pr{2033}, \pr{2034}, \pr{2039}, \pr{2042}, \pr{2045}, \pr{2045},\pr{2049},
  \pr{2053}, \pr{2054}, \pr{2065}, \pr{2076}, \pr{2077}, \pr{2078})
- Adds support for the `MSH` mesh format in the external partitioner (\pr{2029})
- Adds support for the external partitioner in \arcane{ArcaneCaseMeshService}.
  This allows it to be used in mesh tags (\pr{2040})
- Adds support for multi-dimensional meshes in the external partitioner
  (\pr{2048})
- Improves support for order 2 meshes (\pr{2056}, \pr{2060}, \pr{2062},
  \pr{2063})
- Adds support for using the max norm in bit-by-bit comparisons of variables
  (\pr{2068}, \pr{2069}, \pr{2070}, \pr{2071}, \pr{2090}, \pr{2091})
- Adds methods to create and use \arcane{Ref<ISerializeMessage>} instead of an
  \arcane{ISerializeMessage} pointer. This will allow for cleaner management of
  the lifetime of these instances (\pr{2108})
- Makes buffer management thread-safe in \arcane{VariableSynchronizerMng}
  (\pr{2109})
- Adds a tag to variables that will be post-processed in the current iteration.
  This only works for the default %Arcane post-processing module
  (\arcane{ArcanePostProcessingModule}) (\pr{2145}, \pr{2148})
- Adds support for MacOS 14 and 15 (\pr{2151}, \pr{2155}, \pr{2157}, \pr{2164})
- Adds an API for managing windows in shared memory via MPI. This API also works
  in shared memory and hybrid modes. The page \ref arcanedoc_parallel_shmem
  indicates how to use these features (\pr{2158}, \pr{2170}, \pr{2179})
- Adds support for [zstd](https://github.com/facebook/zstd) as an implementation
  of \arcane{IDataCompressor} (\pr{2160})
- Adds the method \arcane{MeshUtils::computeNodeNodeViaEdgeConnectivity()} to
  calculate nodes connected by edges (\pr{2176})
- Adds the possibility of using a class instance as a parameter of
  \arcane{Parallel::BitonicSort} (\pr{2198})
- Adds constructors for \arcanemat{MeshMaterialVariableRef} from an
  \arcanemat{IMeshMaterialMng} (\pr{2200})
- Adds experimental support for `Quad9`, `Hexaedron27`, and `Pyramid13` types
  (\pr{2205}, \pr{2206})
- Adds methods in \arcane{MeshUtils} to directly calculate the owners of nodes,
  edges, and faces when there are no ghost meshes (\pr{2230}, \pr{2236},
  \pr{2237}, \pr{2238})
- Adds the possibility of disabling the entity owner verification test
  (\pr{2239})

### Accelerator API

- Adds \arcane{MultiArray2} accelerator support (\pr{1989}, \pr{1993})
- Uses \arcane{MemoryUtils::getDefaultDataMemoryResource()} for the default
  memory used by \arcane{NumArray} and \arcane{RunQueue}. Previously, unified
  memory was always used. This allows taking into account the environment
  variable `ARCANE_DEFAULT_DATA_MEMORY_RESOURCE` which allows changing the
  default memory resource used (\pr{1997})
- Uses `std::less` for comparison in \arcaneacc{GenericSorter} (\pr{2002})
- Adds \arcaneacc{RunQueueEvent::hasPendingWork()} to know if the
  \arcaneacc{RunQueue} associated with an event is currently running (\pr{2006})
- Adds views for the \arcane{ItemVariableScalarRefT} type (\pr{2098})
- Adds the possibility of changing the execution policy of an
  \arcanemat{IMeshComponent} used to create instances of
  \arcanemat{ComponentCellVector}, \arcanemat{MatCellVector}, and
  \arcanemat{EnvCellVector} associated with it. This mechanism is experimental
  and is activated via the method
  \arcanemat{IMeshComponent::setSpecificExecutionPolicy()}. (\pr{2105})
- Adds support for CUDA 13 (\pr{2183})
- Adds support for ROCM 7 (\pr{2212})
- Adds experimental support for automatically calculating the number of threads
  per block to achieve maximum accelerator occupancy (\pr{2196}, \pr{2197})
- Removes the implementation of reductions using atomic operations (\pr{2214})

### Changes

- Moves classes from %Arccore namespace Arccore to the Arcane namespace. `using`
  statements to the `Arccore` namespace are added to ensure compatibility with
  existing code (\pr{1974}, \pr{1976}, \pr{1977}, \pr{1978}, \pr{1979},
  \pr{1983}, \pr{1984}, \pr{1985})
- Moves \arcane{SerializeMessage} to the internal API of %Arcane (\pr{1995})
- Ensures that the entity group is properly recalculated before calling
  \arcane{ItemGroup::checkIsSorted()} (\pr{2024})
- Automatically adds the `.vtk` extension in the VTK writer (\pr{2055})
- Adds the possibility of allowing non-matching faces in
  \arcane{MeshNodeMerger} (\pr{2074})
- Uses `XML_PARSE_HUGE` in the `libxml2` reader to handle large XML elements
  (more than 10MB) (\pr{2094}, \pr{2097})
- Allows nodes that merge with themselves in \arcane{MeshNodeMerger} (\pr{2106})
- Allows the use of the hyphen character (`-`) in variable names (\pr{2119})
- Allows multiple instances of \arcane{StandaloneSubDomain} (\pr{2127})
- Uses \arcane{ItemLocalId} instead of \arcane{Item} for indexing methods of
  classes deriving from \arcane{MeshPartialVariable} (\pr{2184})

### Corrections

- Does not disable the update of ghost meshes if \arcane{ItemFamilyNetwork} is
  used (\pr{1998})
- Corrects potential 'Read after free' in
  \arcane{ParallelMngDispatcher::setDefaultRunner()} (\pr{2021})
- Corrects the management of \arcane{TimeHistoryMngInternal} when replication is
  active (\pr{2035})
- Increments the timestamp of \arcane{ItemGroup} after reading a protection
  (\pr{2072})
- Corrects the historical writing of curves when the call to
  \arcanemat{ITimeHistoryMng::addValue()} was not made by all ranks of an
  \arcane{IParallelMng}. This resulted in a deadlock (\pr{2118})
- Corrects various issues under Win32 (\pr{2134}, \pr{2141}, \pr{2142},
  \pr{2143},\pr{2146}, \pr{2147}, \pr{2161}, \pr{2163}, \pr{2165}, \pr{2168},
  \pr{2172})
- Corrects the extra written terminal `\0` in protections/recoveries in HDF5
  format (\pr{2144})
- Ensures that data padding is correct when using multiple accelerator
  synchronizations (\pr{2193})

### Internal

- Moves the \arcane{mesh::FaceReorienter} class to `arcane/core` (\pr{1969})
- Adds the \arcane{mesh::ItemsOwnerBuilder} class to calculate entity owners
  without needing a ghost mesh layer. This ensures owner consistency after mesh
  modifications (\pr{2008},\pr{2016}, \pr{2019})
- Cleanup and various improvements in patch Cartesian mesh support (\pr{2031})
- Ensures that the type of elements used in \arcane{NumArray} satisfies the
  `std::is_trivially_copyable` criterion (\pr{2032})
- Cleanup of mesh utility script management (\pr{2043})
- Uses mesh services for the direct execution mechanism (\pr{2046})
- Uses one \arcane{ItemTypeMng} instance per mesh (\pr{2079})
- Improves the management of multi-dimensional or non-manifold meshes
  (\pr{2080}, \pr{2081}, \pr{2083}, \pr{2085}, \pr{2086}, \pr{2089})
- Adds \arcane{Item::hasFlags()} and \arcane{ItemBase::hasFlags()} methods
  (\pr{2087})
- Adds support to make the method \arcane{ItemGroupImpl::_checkNeedUpdate()}
  thread-safe. This mechanism is experimental and is activated by setting the
  environment variable `ARCANE_USE_LOCK_FOR_ITEMGROUP_UPDATE` to `1` (\pr{2110})
- Cleanup and removal of compilation warnings in several files of the
  `arcane_core` component (\pr{2112}, \pr{2114}, \pr{2115}, \pr{2116})
- Various improvements in the python wrapper (\pr{2153}, \pr{2154}, \pr{2156},
  \pr{2159}, \pr{2181})
- Moves the copy and fill methods of \arcane{ConstMemoryView},
  \arcane{MutableMemoryView}, \arcane{ConstMultiMemoryView}, and
  \arcane{MutableMultiMemoryView} to the `MemoryUtils.h` file (\pr{2169},
  \pr{2171})
- Updates the RapidJSON version on commit '24b5e7a8' (\pr{2180})
- Uses `arcane/core` instead of `arcane` for header file paths in Arcane
  (\pr{2187})
- Various improvements in the Neo library (\pr{2210}, \pr{2233}, \pr{2234},
  \pr{2235})
- Optimizes the search for existing faces and edges when using a node hash for
  `uniqueId()` (\pr{2218}, \pr{2219})
- Adds support for Hypre 3.0 (\pr{2228})

### Compilation and Continuous Integration (CI)

- Improves compatibility with Doxygen version 1.10.0 (\pr{2001}, \pr{2003},
  \pr{2014})
- Adds a necessary workflow to apply 'Merge Requests' (\pr{2004})
- Removes workflows using `ubuntu-20.04` because this operating system is no
  longer available on GitHub (\pr{2023})
- Corrects compilation error with oneTBB 2022 (\pr{2050})
- Adds support for `.Net 9` (\pr{2059})
- Adds CMake option `ARCANE_ENABLE_ALEPH` to disable Aleph (\pr{2075})
- Adds IFPEN workflows using C++20 (\pr{2027})
- Removes the workflow using `windows-2019` because it is no longer supported by
  github (\pr{2117})
- Adds the possibility to compile only Arccore from `framework` (\pr{2125})
- Adds preliminary support for compiling with Guix (\pr{2128})
- Adds Win32 workflow in debug mode (\pr{2135})
- Updates the `compile-all-vcpkg` workflow with vcpkg version 2025.06
  (\pr{2140})
- Increases the number of tests in the Win32 workflow (\pr{2162})
- Adds workflow for MacOS 15 and MacOS 26 (\pr{2166}, \pr{2167}, \pr{2188},
  \pr{2192}, \pr{2208})
- Adds Arcane installation in the `compile-all-vcpkg` workflow (\pr{2182})
- Switches to version 5 of `actions/checkout` (\pr{2186})
- Adds option `--parallel 4` for `ctest` in IFPEN 2021 workflows (\pr{2189},
  \pr{2201})
- Updates docker images to add CUDA 13 and CLANG 21 (\pr{2190}, \pr{2191})
- Removes `vtu` tests in the ARM64 workflow for CircleCI because the system VTK
  version contains a bug when reading XML (\pr{2195})

### Arccore

- Moves certain classes from `message_passing_mpi` to the private API of
  %Arccore (\pr{1992})
- Removes the `MpiSerializeMessageList` class which has not been used for a long
  time (\pr{2000})

### Alien

- Removes certain debug information (\pr{1968})
- Removes use of \arcane{BasicSerializeMessage} (\pr{1986})
- Corrects bug in block matrix-vector product (\pr{2044})
- Adds support for AdaptiveCpp 24.10 (\pr{2051}, \pr{2067})
- Adds support for Trilinos version 16.1 (\pr{2088})
- Adds support for Visual Studio 2022 (\pr{2095}, \pr{2096})
- Adds PETSc KSPGMRESSetBreakdownTolerance option (\pr{2178})

___

## Arcane Version 3.15.3 (February 4, 2025) {#arcanedoc_version3150}

### New Features/Improvements

- Adds service to subdivide an initial mesh (\pr{1937}, \pr{1938})
- Starts support for meshes with multiple dimensions and non-manifold meshes
  (\pr{1922}, \pr{1923}, \pr{1931}, \pr{1932}, \pr{1934}, \pr{1935}, \pr{1936},
  \pr{1943}, \pr{1944}, \pr{1945}, \pr{1948})
- Adds experimental support for generating the uniqueId() of faces and edges
  from the uniqueId() of the nodes they comprise (\pr{1851}, \pr{1920})
- Improves support for polyhedral meshes (\pr{1846}, \pr{1847}, \pr{1925})
- Adds the `compact-after-allocate` property to disable mesh compaction upon
  allocation (\pr{1857})

### Accelerator API

- Adds overloads with \arcane{SmallSpan} of \arcane{NumArray} and
  \arcane{MDSpan} methods for dimension 1.
- Allows concurrent creation of \arcaneacc{RunQueue} and
  \arcaneacc{RunCommand} (\pr{1842})
- Does not use the accelerator API for the creation of
  \arcanemat{ComponentItemVector} if multi-threading is active (\pr{1840})
- Uses the accelerator API to initialize \arcanemat{AllCellToAllEnvCell}
  (\pr{1839})
- Automatically calls `cudaMemAdvise()` on regions allocated by
  `cudaMallocManaged()` if the environment variable
  `ARCANE_CUDA_MEMORY_HINT_ON_DEVICE` is set (TODO: document) (\pr{1838})
- Adds SYCL support for two-way list partitioning (\pr{1858})
- Adds multi-threaded versions of the scan
  (\arcaneacc{GenericScan}) and filtering \arcaneacc{GenericFiltering}
  (\pr{1878}, \pr{1880}).
- Adds support for \arcane{ParallelLoopOptions::grainSize()} for
  multi-dimensional loops (\pr{1890})
- Adds specific implementation of \arcaneacc{GenericFiltering} when the input
  and output use the same array (\pr{1882})
- Makes the management of \arcane{TaskFactory} observers thread-safe (\pr{1883})
- Adds support for copying instances of \arcaneacc{RunQueueEvent} with a
  reference semantics (\pr{1895})
- Always makes calls to \arcaneacc{makeQueue()} thread-safe (\pr{1898})
- Removes the implementation of reductions using atomic operations. They are not
  available for all types and are less performant than the implementation using
  a tree (\pr{1908})
- Makes \arcane{StandaloneAcceleratorMng} methods `const` (\pr{1912})
- Various optimizations and refactoring (\pr{1843}, \pr{1845}, \pr{1854},
  \pr{1855}, \pr{1860}, \pr{1875}, \pr{1881}, \pr{1891})

### Changes

- Various refactoring in \`arcane{builtInGetValue()}\` to avoid certain copies
  and use `std::from_chars()` for the `double` type (\pr{1903}, \pr{1909},
  \pr{1910}, \pr{1911}, \pr{1918}, \pr{1919}, \pr{1949}, \pr{1956})
- Moves the methods `isNearlyZero()`, `normL2()`, and `squareNormL2()` from the
  numerical classes \arcane{Real2}, \arcane{Real3}, \arcane{Real2x2}, and
  \arcane{Real3x3} to the namespace Arcane::math (\pr{1871}).
- Renames \arcanemat{ComponentItemLocalId} to \arcanemat{ConstituentItemLocalId}
  and adds two classes \arcanemat{MatItemLocalId} and \arcanemat{EnvItemLocalId}
  to strongly type access to material variables (\pr{1862})
- By default, uses the `mli2` extension instead of `mli` for files generated by
  Lima. The `mli` format is obsolete and should no longer be used (\pr{1849},
  \pr{1850})

### Fixes

- Fixes compilation with ROCM version 5 (\pr{1852})
- Fixes the number of the first `uniqueId()` generated after a restart when the
  mesh is empty, setting it to `0` instead of `1`. This allows for the same
  numbering whether or not a restart occurs in this specific case
  (\pr{1924})
- Fixes memory leak for \arcaneacc{RunCommand} created by \arcaneacc{RunQueue}
  with a priority other than the default priority (\pr{1927})
- Fixes incorrect return values in \arcaneacc{GenericFilterer} and
  \arcaneacc{GenericPartitioner} when input arrays have a zero size (\pr{1929})
- Fixes missing call to `MPI_Group_free` when creating an \arcane{IParallelMng}
  from an \arcane{IParallelMng} (\pr{1933})
- Correctly recalculates synchronization information after the creation of an
  AMR patch (\pr{1946})
- Fixes memory leak in Aleph when creating Hypre, PETSc, or Epetra matrices
  (\pr{1947})
- Fixes array overflow if multi-threading is used with a single thread
  (\pr{1954})
- Fixes memory leaks in certain operations on the DOM (\pr{1965})

### Internal

- Adds class \arcane{mesh::ItemsOnwerBuilder} to calculate the owners of faces
  from the owners of meshes (\pr{1861}, \pr{2082})
- Uses \arcane{IMesh} instead of \arcane{mesh::DynamicMesh} in
  \arcane{mesh::MeshExchangeMng} (\pr{1841})
- Adds specific `float` type for HDF5 utilities (\pr{1837})
- Separates the explicit instantiation of certain classes into multiple files to
  speed up compilation (\pr{1836}, \pr{1873})
- Starts support for the types \arcane{Float128}, \arcane{Int128},
  \arcane{Float16}, \arcane{Float32}, \arcane{BFloat16}, and \arcane{Int8}
  (\pr{1835}, \pr{1863}, \pr{1864}, \pr{1866}, \pr{1867}, \pr{1868}, \pr{1869},
  \pr{1870}, \pr{1914})
- Uniformizes multi-threaded profiling management (\pr{1877})
- Implements \arccore{IMemoryAllocator3} instead of \arccore{IMemoryAllocator}
  for \arcane{SmallArray} (\pr{1889})
- Uses dependency injection to create the services managing threads in
  \arcane{Application} (\pr{1900}, \pr{1902})
- Removes the obsolete method \arcane{mesh::DynamicMesh::addFace()} (\pr{1930})
- Adds observables to ensure that views on internal variables in
  \arcane{mesh::ItemFamily} are always valid following an external change
  (\pr{1953})

### Build and Continuous Integration (CI)

- Removes the possibility of installing %Arccore separately from %Arcane. This
  ensures that %Arccore and %Arcane will always be consistent (\pr{1865}).
- Simplifies component dependency management to speed up compilation
  (\pr{1874}).
- Updates the `compile-all-vcpkg` workflow with version 2024.12
  (\pr{1894}).
- Adds `compile-all-vcpkg` workflow with ubuntu 24.04 (\pr{1896})
- Uses `.Net 8` for certain `compile-all-vcpkg` workflows and adds C# wrapper
  tests (\pr{1897})
- Adds workflow for ROCM version 6.3.1 and 5.7.1 (\pr{1899})
- Uses `.mli2` instead of `.mli` for Lima mesh files (\pr{1904})
- Updates CI images (\pr{1905}, \pr{1915})
- Adds workflow with `-fsanitize=address` (\pr{1940}, \pr{1955})

### Arccore

- Moves %Arccore classes from the Arccore namespace to the Arcane namespace
  (\pr{1963}, \pr{1966})
- Removes obsolete methods of \arccore{IMemoryAllocator} (\pr{1959})

### Alien

- Removes use of `boost::remove_reference` and replaces it with
  `std::remove_reference` (\pr{1844})
- Fixes compilation when PETSc is compiled with MUMPS support (\pr{1917})

___

## Arcane Version 3.14.15 (December 11, 2024) {#arcanedoc_version3140}

### New Features/Improvements

- Adds the possibility of choosing the version used for calculating face IDs in
  the dataset (\pr{1826})
- Adds experimental support for reading MSH files version 4.1 in binary format
  (\pr{1824})
- Adds a mechanism to delete ghost meshes from refined meshes (\pr{1716},
  \pr{1785}, \pr{1818})
- Adds methods \arcane{ICartesianMesh::coarseZone2D()} and
  \arcane{ICartesianMesh::coarseZone3D()} to define a block of an AMR patch
  (\pr{1697})
- Adds support in patch-based AMR to define the initial mesh (\pr{1678},
  \pr{1774})
- Adds the possibility of choosing the face numbering version in the dataset
  (\pr{1674})
- Adds a new implementation of the associative array
  \arcane{impl::HashTableMap2}. This implementation is currently internal to
  Arcane. The new implementation is faster and consumes less memory than the old
  (\arcane{HashTableMapT}). It also has an API compatible with
  `std::unordered_map` (\pr{1638}, \pr{1639}, \pr{1640}, \pr{1650})
- Adds support for MPI/IO block writing in the writer `VtkHdfV2PostProcessor`
  (\pr{1648}, \pr{1649})
- Adds support for polyhedral meshes (\pr{1619}, \pr{1620}, \pr{1496},
  \pr{1746}, \pr{1747}, \pr{1748}, \pr{1761}, \pr{1762}, \pr{1795}, \pr{1816},
  \pr{1829})
- Adds utility method \arcane{MeshUtils::computeNodeNodeViaEdgeConnectivity()}
  to create node-to-node connectivities via edges (\pr{1614})
- Adds a mechanism to verify that all ranks synchronize the same variable. This
  is done by setting the environment variable
  `ARCANE_CHECK_SYNCHRONIZE_COHERENCE` (\pr{1604})
- Adds support for setting the debug name for \arcane{NumArray} (\pr{1590})
- Adds the possibility of displaying the call stack via the debugger when a
  signal (SIGSEGV, SIGBUS, ...) is received (\pr{1573})
- Adds a new version of initial coarsening that preserves the initial numbering
  and guarantees the same numbering regardless of the decomposition (\pr{1557})
- Adds implementation of \arcane{IParallelMng::scan()} for shared memory and
  hybrid mode (\pr{1548})

### Accelerator API

- Adds method \arcaneacc{Runner::deviceMemoryInfo()} to retrieve free memory and
  total memory of an accelerator (\pr{1821})
- Displays more properties when describing the used accelerator (\pr{1819})
- Adds the possibility of changing the memory resource associated with
  \arcane{MemoryUtils::getDefaultDataAllocator()} (\pr{1808})
- Uses `const RunQueue&` or `const RunQueue*` instead of `RunQueue&` and
  `RunQueue*` for certain method arguments that use \arcaneacc{RunQueue}
  (\pr{1798})
- Prohibits using the same \arcaneacc{RunCommand} instance twice. It is
  temporarily possible to allow this by setting the environment variable
  `ARCANE_ACCELERATOR_ALLOW_REUSE_COMMAND` to `1` (\pr{1790})
- Adds class \arcaneacc{RegisterRuntimeInfo} to pass arguments for accelerator
  runtime initialization (\pr{1766})
- Adds methods to cleanly retrieve the native implementation corresponding to
  \arcaneacc{RunQueue} and deprecates \arcaneacc{RunQueue::platformStream()}
  (\pr{1763})
- Adds header files for advanced algorithms whose name is identical to the class
  name (\pr{1757})
- Adds implementation of RUNCOMMAND_MAT_ENUMERATE() for \arcanemat{AllEnvCell}
  (\pr{1754})
- Adds methods \arcaneacc{IAcceleratorMng::runner()} and
  \arcaneacc{IAcceleratorMng::queue()} which return instances instead of
  pointers to \arcaneacc{Runner} and \arcaneacc{RunQueue}
  (\pr{1752})
- Makes the construction methods of \arcaneacc{RunQueue} and
  \arcaneacc{RunCommand} private. Instances of these classes must be created via
  \arcaneacc{makeQueue} or \arcaneacc{makeCommand} (\pr{1752})
- Various optimizations in the update of constituents
  (\arcanemat{MeshMaterialModifier}) (\pr{1559}, \pr{1562}, \pr{1679},
  \pr{1681}, \pr{1682}, \pr{1683}, \pr{1687}, \pr{1689}, \pr{1690}, \pr{1691},
  \pr{1704}, \pr{1720}, \pr{1729}, \pr{1731}, \pr{1733}, \pr{1738}, \pr{1739},
  \pr{1741}, \pr{1742}, \pr{1831})
- Adds class \arcaneacc{ProfileRegion} to specify a region for profiling on the
  accelerator (\pr{1695}, \pr{1734}, \pr{1768})
- Adds an internal class \arcane{impl::MemoryPool} to keep a list of allocated
  blocks. This mechanism currently only works with the CUDA implementation. It
  is not active by default (see \ref arcanedoc_acceleratorapi_memorypool)
  (\pr{1684}, \pr{1685}, \pr{1686}, \pr{1699}, \pr{1703}, \pr{1724}, \pr{1725},
  \pr{1726}, \pr{1776})
- Adds partitioning algorithm \arcaneacc{GenericPartitioner} (\pr{1713},
  \pr{1717}, \pr{1718}, \pr{1721}, \pr{1722})
- Uniformizes accelerator algorithm constructors to take an \arcaneacc{RunQueue}
  as an argument (\pr{1714}
- Uses the accelerator API for the creation of \arcanemat{EnvCellVector} and
  \arcanemat{MatCellVector} (\pr{1710}, \pr{1711})
- Allocates the memory of \arcane{ItemVector} via UVM. This allows elements of
  this class to be accessible on accelerators (\pr{1709})
- Adds sorting algorithm \arcaneacc{GenericSorter} (\pr{1705}, \pr{1706})
- Adds the possibility of using host memory for the return value of
  \arcaneacc{GenericFilterer} (\pr{1701})
- Adds explicit synchronization for \arcane{DualUniqueArray} (\pr{1688})
- Adds the possibility of retrieving the `backCell()` and `frontCell()` of a
  face on the accelerator (\pr{1607})
- Adds accelerator implementation of
  \arcanemat{IMeshMaterialMng::synchronizeMaterialsInCells()}. This
  implementation is activated if the environment variable
  `ARCANE_ACC_MAT_SYNCHRONIZER` is set to `1` (\pr{1584})
- Adds the possibility of using the ATS mechanism with CUDA for allocation
  (\pr{1576})
- Ensures that messages displayed by the CUPTI integration are not split in
  multi-threading (\pr{1571})
- Adds the possibility of displaying and interrupting profiling
  (\pr{1561}, \pr{1569})
- Adds macro RUNCOMMAND_SINGLE() to execute an accelerator command for a single
  iteration (\pr{1565})
- By default, allocates managed memory (UVM) on a multiple of the system page
  size (\pr{1564})
- Adds detection and display of 'Page Faults' in the CUPTI integration
  (\pr{1563})
- Automatically chooses the device associated with an MPI rank on a node
  according to a round-robin mechanism (\pr{1558})
- Adds support for compiling for CUDA with the Clang compiler (\pr{1552})

### Changes

- Renames \arcane{NumArray::resize()} to \arcane{NumArray::resizeDestructive()}
  and \arcane{NumArray::fill()} to \arcane{NumArray::fillHost()} (\pr{1809})
- Moves the platform methods managing allocators to \arcane{MemoryUtils} and
  deprecates some of them (\pr{1806}, \pr{1817})
- In the \arcane{ArcaneCaseMeshService} service, initializes the variables
  specified in the dataset after applying partitioning instead of before
  (\pr{1751})
- Uses all ranks for parallel reading of GMSH files. This prevents having empty
  partitions later, which is not supported by ParMetis (\pr{1735})
- Deprecates \arcane{NumArray::span()} and \arcane{NumArray::constSpan()}
  \pr{1723}
- Removes the obsolete class `Filterer` (\pr{1708})
- Makes the constructors of variables that take a \arcane{VariableBuildInfo} as
  an argument explicit (\pr{1693})
- Changes the default behavior of \arcane{ILoadBalanceMng} so that it does not
  take into account the number of allocated variables, thus resulting in
  partitioning that depends only on the mesh (\pr{1673})
- Removes post-processing in `EnsighHdf` format. This format was experimental
  and is no longer available in recent versions of Ensight (\pr{1668})
- Writes edges and associated connectivities in
  \arcane{MeshUtils::writeMeshConnectivity()} (\pr{1651})
- Deprecates \arcane{mesh::ItemFamily::infos()} (\pr{1647})
- Raises an exception if an already allocated mesh is attempted to be read
  (\pr{1644})
- Deprecates \arcane{Item::internal()} (\pr{1627}, \pr{1642})
- Deprecates \arcane{IItemFamily::findOneItem()} (\pr{1623})
- Raises an exception (instead of a warning) if the user-specified partitioner
  is not available (\pr{1635})
- Removes the possibility of compiling with `mono` (\pr{1583})
- Deprecates version 1 of \arcane{CartesianMeshCoarsening} (\pr{1580})

### Fixes

- Uses the same values for \arccore{ISerializer::eDataType} and
  \arcane{eDataType} (\pr{1827}, \pr{1828})
- Fixes compilation error with CRAY MPICH (\pr{1778})
- Fixes compilation with libxml2 versions 2.13+ (\pr{1715})
- Correctly positions the communicator of
  \arccore{MessagePassing::MessagePassingMng} associated with the sequential
  implementation \arcane{SequentialParallelMng}. Previously, this was only done
  for the MPI implementation (\pr{1661})
- In the VTK reader, keeps the variables read from the file after the reader is
  destroyed (\pr{1655})
- Does not destroy the instance used in
  \arcane{ArcaneLauncher::setDefaultMainFactory()} at the end of execution.
  This instance is not necessarily allocated via `new` (\pr{1643})
- Fixes incorrect behavior in subdomain management with new connectivities
  (\pr{1636})
- Fixes non-consideration of the environment variable
  `ARCANE_DOTNET_USE_LEGACY_DESTROY` (\pr{1570})
- Adds missing call to \arcane{ArcaneMain::arcaneFinalize()} when using `.Net`
  (\pr{1567})
- Fixes incorrect handling of options for the tool `arcane_partition_mesh`
  (\pr{1555})

### Internal

- Adds environment variable `ARCANE_CUDA_MEMORY_HINT_ON_DEVICE`
  to automatically call `cudaMemAdvise()` on unified memory to force it to use a
  specific allocator (\pr{1833})
- Calls \arccore{Array::resizeNoInit()} instead of \arccore{Array::resize()}
  when resizing material variables. Initialization is done later, which prevents
  double initialization (\pr{1832})
- Renames `Adjency` to `Adjacency` in certain classes and methods (\pr{1823})
- Adds command line option `-A,UseAccelerator=1` to specify that accelerator
  execution is desired. The backend used will be automatically chosen based on
  the compilation configuration (\pr{1815})
- Adds typedef `AlephInt` which is used to specify the indices of rows and
  columns of matrices and vectors. Currently, this type is `int`, but when
  64-bit support is active, it will be
  `long int` or `long long int` (\pr{1770})
- Frees serialization buffers as soon as possible during load balancing
  (\pr{1744}, \pr{1756})
- Adds an experimental service allowing mesh subdivision during initialization
  (\pr{1606}, \pr{1728})
- Makes the classes \arcane{ItemBase} and \arcane{MutableItemBase} public
  \pr{1740}
- Adds an internal finalization method for the accelerator API. This allows
  statistics to be displayed and associated resources to be freed
  (\pr{1727})
- Adds tests for using multiple criteria with multi-mesh partitioning
  (\pr{1719}, \pr{1772})
- In \arcane{BitonicSort}, does not allocate arrays for ranks and indices if
  they are not used (\pr{1680})
- Uses a new hashing implementation for `ItemInternalMap`. This implementation
  is active by default, but the old one can be used by setting the option
  `ARCANE_USE_HASHTABLEMAP2_FOR_ITEMINTERNALMAP` to `OFF` during configuration
  (\pr{1611}, \pr{1617}, \pr{1622}, \pr{1624}, \pr{1625}, \pr{1628}, \pr{1629},
  \pr{1631}, \pr{1677}, \pr{1745})
- Cleanup and refactoring of partitioning with ParMetis to use
  \arcane{IParallelMng} instead of calling MPI directly (\pr{1662}, \pr{1665},
  \pr{1667}, \pr{1671})
- Adds support for creating a sub-communicator in the style of
  `MPI_Comm_Split` (\pr{1669}, \pr{1672})
- Cleanup and refactoring of classes managing the numbering of
  `uniqueId()` for edges (\pr{1658})
- Uses a pointer to \arcane{mesh::DynamicMeshKindInfos} instead of an instance
  in \arcane{mesh::ItemFamily} (\pr{1646})
- Adds to \arcane{ICaseMeshService} the possibility of performing operations
  after partitioning (\pr{1637})
- Adds internal API for \arcane{IIncrementalItemConnectivity}
  (\pr{1615}, \pr{1618}, \pr{1626})
- Adds a `build` type entry point for the unit test service (\pr{1613})
- Adds the possibility in \arcane{IIncrementalItemConnectivity} to notify of the
  addition of multiple entities at once (\pr{1610})
- Adds a method to retrieve the call stack via LLVM or GDB. This allows, in
  particular, obtaining line numbers in the call stack
  (\pr{1572}, \pr{1597}, \pr{1616}) TODO INDICATE METHOD
- Adds experimental support for KDI (\pr{1594}, \pr{1595}, \pr{1599})
- Various optimizations in the `VtkHdfV2PostProcessor` service (\pr{1575},
  \pr{1600})
- [EXPERIMENTAL] Adds a new version of face `uniqueId()` calculation based on
  the `uniqueId()` of the nodes that compose it. This only works in sequential
  mode for now (\pr{1550})
- Adds function to generate a `uniqueId()` from a list of
  `uniqueId()` (\pr{1549})
- Optimizes the calculation of the version 3 of ghost meshes (\pr{1547})

### Build and Continuous Integration (CI)

- Uses a `dotnet` wrapper for compilation. This wrapper ensures that the user's
  HOME is not modified, which could cause locking issues when multiple instances
  of `dotnet` run simultaneously (\pr{1789}, \pr{1791}, \pr{1792}, \pr{1830})
- Adds the possibility of adding libraries during the linking of
  `arccore_message_passing_mpi`. This ensures that certain libraries are
  properly added to the link and is notably used for MPI 'GPU-Aware' support
  with CRAY MPICH (\pr{1786})
- Adds 'ubuntu 22.04' workflow for IFPEN dockers (\pr{1781})
- Adds CMake variable `ARCCON_NO_TBB_CONFIG` to force not to use the CMake
  configuration file for TBB (\pr{1779})
- Adds protection/restart tests for coarsening (\pr{1707})
- Fixes compilation error when PETSc is not compiled with MUMPS (\pr{1694})
- Adds VTK reader test with properties (\pr{1656}, \pr{1659})
- Adds support for a connectivity checksum in mesh tests (\pr{1654})
- Writes test output files to the test directory to allow them to be run in
  parallel (\pr{1541}, \pr{1653})
- Updates IFPEN images 2021 (\pr{1542}, \pr{1579}, \pr{1587}, \pr{1588},
  \pr{1592}, \pr{1593}, \pr{1598})
- Removes internal versions of `hostfxr.h` and
  `coreclr_delegates.h`. These files are now in the dotnet SDK (\pr{1591})
- Fixes FlexLM detection and configuration (\pr{1602}, \pr{1630})
- Removes test output directories after execution to reduce storage footprint
  (\pr{1581})
- Runs CI tests in parallel for multiple workflows (\pr{1553})
- Refactors the CI action system and images to make it more flexible (\pr{1545})

### Arccore

- Adds conversions of \arccore{SmallSpan<std::byte>} to and from
  \arccore{SmallSpan<DataType>} {\pr{1731})
- Fixes incorrect value for the size passed as an argument to
  \arccore{IMemoryAllocator::deallocate()} (\pr{1702})
- Adds \arcaneacc{RunQueue} argument for \arccore{MemoryAllocationArgs} and
  \arccore{MemoryAllocationOptions}. This is not used for now but will allow for
  specific allocations to a run queue later (\pr{1696})
- Adds support for specific allocators for \arccore{SharedArray} (\pr{1692})
- Makes the constructors and destructors of \arccore{MemoryAllocationOptions}
  `inline` (\pr{1664})
- Adds method to set the communicator associated with
  \arccore{MessagePassing::MessagePassingMng} (\pr{1660})

### Axlstar

- Replaces `std::move()` with `std::forward()` in the generation of certain
  methods (\pr{1773})
- Generates the methods to retrieve the functions associated with dataset
  options via the specified interface (\pr{1601})
- Removes the date from generated files so they are not modified if regenerated
  identically (\pr{1797}

### Alien

- Adds a plugin based on the library
  [composyx](https://gitlab.inria.fr/composyx/composyx) (\pr{1801})
- Uses a different output directory for each test so that they can be run in
  parallel (\pr{1775})
- Fixes listing outputs for certain tests (\pr{1765})
- Fixes listing outputs for the IFPSolver backend (\pr{1730})
- Fixes runtime error when using the sequential implementation of
  \arcane{IParallelMng} (\pr{1666})
- Uses a unique filename for test output files. This allows them to be run in
  parallel (\pr{1663})
- Retrieves the MPI communicator via
  \arccore{MessagePassing::MessagePassingMng::communicator()} (\pr{1657})
- Adds accelerator support for certain parts (\pr{1632}, \pr{1634})
- Fixes initialization with recent versions (2.27+) of Hypre (\pr{1603})
- Adds support for the SPAI solver via PETSc (\pr{1578})
- Adds support for 'Matrix Market' format output with PETSc (\pr{1577})

___

## Arcane Version 3.13.08 (July 19, 2024) {#arcanedoc_version3130}

### New Features/Improvements

- Adds experimental support to create a mesh variable without reference. This
  requires initializing it with an instance of \arcane{NullVariableBuildInfo}
  (\pr{1510}).
- Adds support for dynamic load balancing with multiple meshes (\pr{1505},
  \pr{1515}).
- Adds support for synchronizations on a subset of ghost entities (\pr{1468},
  \pr{1484}, \pr{1486}).
- README improvement and translation to English (\pr{1466}).
- Adds experimental patch AMR support (\pr{1413}).
- Starts support for integrating Python code during execution. This support is
  considered experimental (\pr{1447}, \pr{1449}, \pr{1454}, \pr{1456},\pr{1461},
  \pr{1462}, \pr{1471}, \pr{1479}, \pr{1493}, \pr{1494}, \pr{1499}, \pr{1501},
  \pr{1502}, \pr{1513}, \pr{1522}, \pr{1525}).

#### Accelerator API

- Adds accelerator access to certain methods (\pr{1539})
- Optimizes calculation kernels for material update management (\pr{1421},
  \pr{1422}, \pr{1424}, \pr{1426}, \pr{1431}, \pr{1432}, \pr{1434}, \pr{1437},
  \pr{1440}, \pr{1441}, \pr{1443}, \pr{1458}, \pr{1472}, \pr{1473}, \pr{1474},
  \pr{1488}, \pr{1489})
- Documentation update (\pr{1483}, \pr{1492}, \pr{1508})
- Adds template arguments to specify the size of integers used for indexing
  (`Int32` or `Int64`) (\pr{1398}).
- Adds support for version 2 reductions in RUNCOMMAND_MAT_ENUMERATE()
  (\pr{1390}).
- Continues SYCL porting work (\pr{1389}, \pr{1391}, \pr{1393}, \pr{1396}).
- Improves SIMD padding management for \arcane{ItemGroup} to only do it on
  demand and do it on the accelerator if possible (\pr{1405}, \pr{1523},
  \pr{1524}).

### Changes

- Removes the component managing vector expressions on variables as it has not
  been used for a long time (\pr{1537})
- Uses `clock_gettime()` instead of `clock()` to measure CPU time used
  (\pr{1532})
- By default, uses the parallel version of the *MSH* reader. It is always
  possible to use the sequential version by setting the environment variable
  `ARCANE_USE_PARALLEL_MSH_READER` to `0` (
  \pr{1528}).
- Removes support for C# compilation with `mono`. `dotnet` (version 6 or higher)
  must now be used (\pr{1470}).
- Renames \arcane{ArrayIndex} to \arcane{MDIndex} (\pr{1397}).
- Adds the possibility of retrieving an \arcanemat{ComponentCell} from an
  \arcanemat{ComponentItemVectorView} (\pr{1478})

### Fixes

- Fixes various bugs in mesh graph management (\pr{1536}, \pr{1535})
- Fixes potential concurrency issues in multi-threaded mode (\pr{1467},
  \pr{1534}, \pr{1533}, \pr{1529})
- Adds support for 1D meshes in the VtkHdfV2 format (\pr{1519})
- Re-enables signal sending in a signal handler. This had been implicitly
  disabled when using `sigaction` instead of
  `sigset` to position signals (\pr{1518})
- Fixes reading groups in the `MSH` format when there are multiple groups per
  physical entity (\pr{1507}, \pr{1509}).
- Adds verification tests in the use of partial variables (\pr{1485})
- Explicitly indicates the underlying type for \arcane{eDataType} (\pr{1418})
- Fixes memory reuse issue for reductions introduced during the creation of
  version 2 of reductions (\pr{1439}).
- Fixes potential incorrect calculation of the number of entities to serialize,
  which could result in an array overflow (\pr{1423}).
- Does not keep behavioral change properties in \arcanemat{MeshMaterialModifier}
  to avoid inconsistencies (\pr{1453}).

### Internal

- Uses a single instance of \arcane{DofManager} (\pr{1538})
- Removes multiple uses of Glib (\pr{1531})
- Removes certain obsolete uses of \arcane{ItemInternal} in the C# wrapper
  (\pr{1517}, \pr{1520})
- Improves C# compilation and SWIG dependencies management by using a single
  project and fixes various dependency issues in CMake management (\pr{1433},
  \pr{1407}, \pr{1410}, \pr{1411}, \pr{1412}, \pr{1414}, \pr{1425}, \pr{1427},
  \pr{1428}, \pr{1429}, \pr{1455}, \pr{1480}, \pr{1487}, \pr{1495}, \pr{1497},
  \pr{1498})
- Various improvements in polyhedral mesh management (\pr{1435}, \pr{1436},
  \pr{1438}, \pr{1463})

### Continuous Integration (CI)

- Adds Python tests in certain workflows (\pr{1448}, \pr{1526}).
- Updates Ubuntu images (\pr{1417}, \pr{1442}, \pr{1503}).
- Activates `ccache` for the `codecov` workflow (\pr{1464}).
- Switches to `vcpkg` version 2024.04 for the `compile-all-vcpkg` workflow
  (\pr{1450}).
- Uses a specific CMake variable to enable test coverage (\pr{1444}, \pr{1445}).
- Updates CI to no longer use obsolete GitHub actions (\pr{1416}).
- Updates CI for IFPEN dockers (\pr{1402}, \pr{1409})
- Adds the possibility of using accelerator tests via googletest (\pr{1401},
  \pr{1403}).
- Explicitly uses the sequential message exchange service for sequential tests
  (\pr{1430}).

### Arccore

- Adds the possibility to specify the memory location of the allocated area in
  \arccore{AbstractArray} (\pr{1803}, \pr{1804})
- Uses \arccore{MessagePassing::MessageRank::anySourceRank()} instead of the
  default constructor to specify
  `MPI_ANY_SOURCE`. The old mechanism remains temporarily valid (\pr{1511},
  \pr{1512}).
- Removes `const` from the return type of certain \arccore{ArrayIterator}
  methods to comply with C++ standards (\pr{1392})

### Axlstar

- Improvement of documentation generated from AXL files (\pr{1452})

### Alien

- Various fixes (\pr{1395}, \pr{1415}, \pr{1465}, \pr{1491}, \pr{1500},
  \pr{1506}, \pr{1514})
- Adds support for infinite norm (\pr{1504})

___

## Arcane Version 3.12.18 (May 02, 2024) {#arcanedoc_version3120}

\note With this version, it is necessary to enable C++20 to use the accelerator
API. To do this, you must set the CMake variable `ARCCORE_CXX_STANDARD=20`
during configuration. You must then have at least GCC 11, Clang 16, or Visual
Studio 2022, as well as CMake version 3.26.

### New Features/Improvements

- Adds the possibility to sort connected faces and edges to nodes by increasing
  uniqueId() (\pr{990}). This is not active by default for compatibility with
  existing code. Sorting allows for the same traversal order for faces and node
  edges, which can help ensure reproducible calculations. To activate it, set
  the environment variable
  `ARCANE_SORT_FACE_AND_EDGE_OF_NODE` to `1` or use the following code:
  ~~~{.cpp}
  Arcane::IMesh* mesh = ...;
  mesh->nodeFamily()->properties()->setBool("sort-connected-faces-edges",true);
  mesh->modifier()->endUpdate();
  ~~~
- Allows modifying the parent patch number during renumbering via
  \arcane{CartesianMeshRenumberingInfo::setParentPatch()} (\pr{986}).
- Adds support for incremental hash function calculation (\pr{983}, \pr{984}).
- Adds implementation of the `SHA1` hash algorithm (\pr{982}).
- Adds data initialization policy to have the same behavior as previous %Arcane
  versions before version 3 (\pr{1017}).
- Adds \arcane{Vector2} and \arcane{Vector3} classes to manage pairs and
  triplets of arbitrary types (\pr{1066}, \pr{1075})
- Adds \arcane{FixedArray} class for fixed-size arrays. This class is similar to
  `std::array` but initializes its values by default and allows for array
  overflow detection (\pr{1063})
- Adds support for 3D de-refinement and renumbering of mesh entities during
  subsequent refinement (\pr{1061}, \pr{1062})
- Adds conversion of \arcanemat{ComponentCell} to \arcanemat{MatCell},
  \arcanemat{EnvCell}, and \arcanemat{AllEnvCell} (\pr{1051}).
- Adds support for saving variable 'hashes' to quickly compare if values are
  different (\pr{1142}, \pr{1143}, \pr{1145}, \pr{1147}, \pr{1150}, \pr{1152},
  \pr{1155}, \pr{1156}, \pr{1158}, \pr{1159}, \pr{1160})
- Adds parallel reading of MSH format files (\pr{1126}, \pr{1136}, \pr{1137},
  \pr{1138}, \pr{1139})
- Refactoring of \arcane{ITimeHistoryMng} management to easily handle multiple
  meshes (\pr{1203}, \pr{1249}, \pr{1256}, \pr{1260}, \pr{1369})
- Adds experimental support for Cartesian patch AMR by duplicating nodes and
  faces (\pr{1167}, \pr{1337}, \pr{1350}, \pr{1351}, \pr{1353})
- Adds a new method (method 4) for AMR patch renumbering to maintain the same
  numbering between a refined mesh, a de-refined mesh, and then a refined mesh
  again (\pr{1108}, \pr{1109})
- Activates 3D node merging (\pr{1361})
- Adds support for XML file indentation via LibXml2 during outputs (\pr{1379})

#### Accelerator API

- Requires C++20 for the accelerator API (\pr{1020}, \pr{1026}, \pr{1030},
  \pr{1050}).
- **INCOMPATIBILITY**: Adds a template argument in \arcane{ExtentsV},
  \arcane{ArrayIndex} allowing specification of the index type. Currently, only
  `Int32` is supported, but `Int64` and `Int16` will be available later. This
  change prevents compiling code that uses the old class value of
  \arcane{ExtentsV}. Since this class is normally internal to %Arcane, this
  should not affect many usages (\pr{1383})
- Adds an asynchronous `fill()` method for mesh variables
  (\arcane{ItemVariableArrayRefT} and \arcane{ItemVariableScalarRefT})
  (\pr{991})
- Adds \arcaneacc{RunQueue::ScopedAsync} class to temporarily make an execution
  queue asynchronous (\pr{978}).
- Optimizes memory management of the \arcaneacc{Filterer} class and adds
  overloads (\pr{1022}, \pr{1023}, \pr{1034}, \pr{1043}, \pr{1210}, \pr{1225},
  \pr{1271})
- Adds support for atomic operations (\pr{1028}, \pr{1032}, \pr{1033})
- Adds support in Aleph for Hypre versions compiled with accelerator support
  (\pr{1125})
- Adds experimental support for automatic pre-fetching of view values
  (\pr{1179}, \pr{1180}, \pr{1304})
- Adds direct reduction algorithm via the \arcaneacc{GenericReducer} class
  (\pr{1306}, \pr{1307})
- Adds list partitioning algorithm via the \arcaneacc{GenericPartitioner}
  class (\pr{1217})
- Makes certain mathematical operations accessible on accelerators (\pr{1294})
- Adds overloads for the `PrefixSum` algorithm ( \pr{1253}, \pr{1254})
- Adds views on partial variables (\pr{1299}, \pr{1308}, \pr{1311}, \pr{1313})
- Uses the accelerator API when updating materials via
  \arcanemat{MeshMaterialModifier} (\pr{1182}, \pr{1183}, \pr{1185}, \pr{1186},
  \pr{1196}, \pr{1204}, \pr{1205},\pr{1211}, \pr{1219}, \pr{1223}, \pr{1224},
  \pr{1226}, \pr{1227}, \pr{1230}, \pr{1233}, \pr{1235}, \pr{1238}, \pr{1239},
  \pr{1241}, \pr{1243}, \pr{1247}, \pr{1257}, \pr{1258}, \pr{1263}, \pr{1268},
  \pr{1283}, \pr{1284}, \pr{1285}, \pr{1287}, \pr{1292}, \pr{1295}, \pr{1312},
  \pr{1347})
- Uses the accelerator API for adding and removing entities in an
  \arcane{ItemGroup} (\pr{1288},\pr{1289}, \pr{1293}).
- Adds a default constructor for \arcaneacc{RunQueue}. In this case, the
  instance is not usable until it has been assigned (\pr{1282})
- Adds the possibility of copying \arcaneacc{RunQueue} with a reference
  semantics (\pr{1221})
- Adds support for choosing a default memory resource for an
  \arcaneacc{RunQueue} (\pr{1278})
- Adds mechanisms to facilitate memory management (\pr{1273}, \pr{1274})
- Adds the possibility of creating a view on data by specifying an instance of
  \arcaneacc{RunQueue} (\pr{1269}).
- Adds conversion methods from \arcane{NumArray} and \arcane{MDSpan} to
  \arccore{SmallSpan} (\pr{1262})
- Adds support for retrieving the iterator index for RUNCOMMAND_ENUMERATE() and
  RUNCOMMAND_MAT_ENUMERATE() (\pr{1270}, \pr{1272})
- Adds support in RUNCOMMAND_MAT_ENUMERATE() for \arcanemat{MatCell} (\pr{1184})
- Adds the function \arcane{VariableUtils::markVariableAsMostlyReadOnly()} to
  mark variables as primarily read-only (\pr{1206})
- Allows the use of \arcane{eMemoryRessource::HostPinned} memory when the
  accelerator runtime is not defined. In this case,
  \arcane{eMemoryRessource::Host} is used (\pr{1315})
- Adds the method \arcane{MemoryUtils::getDeviceOrHostAllocator()} to retrieve
  an allocator on the accelerator if an accelerator runtime is available, and on
  the host otherwise (\pr{1364})
- Starts support for a backend with the SYCL API. This backend is not yet
  functional and is only available for internal tests (\pr{1318}, \pr{1319},
  \pr{1320}, \pr{1323}, \pr{1324}, \pr{1330},\pr{1334}, \pr{1345}, \pr{1355},
  \pr{1363}, \pr{1365}, \pr{1373}, \pr{1374}, \pr{1380}, \pr{1381})
- Starts a new implementation of reductions that allow supporting the SYCL API
  (\pr{1366}, \pr{1368}, \pr{1371}, \pr{1372}, \pr{1377})

### Changes

- **INCOMPATIBILITY**: Removes most `using` statements for mathematical
  functions in the `StdHeader.h` file (\pr{1370}, \pr{1384}).
- Removes the internal class `ComponentItemInternal`, which is replaced by
  `ComponentItemBase` (\pr{1039}, \pr{1053}, \pr{1059}, \pr{1172}, \pr{1181},
  \pr{1187}, \pr{1190}, \pr{1191}, \pr{1192}, \pr{1193}, \pr{1194},
  \pr{1195}, \pr{1197}, \pr{1199}, \pr{1200}, \pr{1335})
- Always activates incremental connectivities (\pr{1166})
- Deprecates \arcane{NumArray::s()}. Use `operator()` or `operator[]` instead
  (\pr{1035})
- Does not initialize the MPI runtime if the message exchange service
  (`MessagePassingService`) is `Sequential` (\pr{1029}).
- Does not set the CMake variable `ARCANE_BUILD_TYPE` by default during
  configuration (\pr{1004}).
- No longer filters signed assemblies when loading C# plugins (\pr{1114})
- Uses \arccore{String} instead of `const String&` for return values of
  \arcane{IVariable} and \arcane{VariableRef} (\pr{1134})
- By default, uses the new version of serialization message list management
  (\pr{1113})
- Uses \arccore{Int16} instead of \arcane{Int8} for the accelerator ID
  (\arcaneacc{DeviceId}) (\pr{1276})
- By default, uses version 2 of entity list management by type in groups
  (\arcane{ItemGroup}. This is used, for example, in
  \arcane{ItemGroup::applyOperation()}. The new version no longer uses subgroups
  but only lists and does not duplicate memory if all entities in the group are
  of the same type (\pr{1174})
- In views, replaces \arcane{ViewSetter} with \arcane{DataViewSetter} and
  \arcane{ViewGetterSetter} with \arcane{DataViewGetterSetter} (\pr{1040})

### Fixes

- Fixes memory leak with asynchronous commands when
  \arcaneacc{RunQueue::barrier()} is not called. This should normally not happen
  because `barrier()` must always be called after an asynchronous command. Now,
  if commands are active when the \arcanacc{RunQueue} is destroyed, an implicit
  barrier is performed (\pr{995})
- Fixes incorrect return type of
  \arcaneacc{MatItemVariableScalarOutViewT::operator[]()} which did not allow
  modifying the value (\pr{981}).
- Fixes incorrect property consideration when creating material variables in
  certain cases (\pr{1012}).
- Fixes incorrect value for the Z dimension for 3D de-refinement (\pr{1074})
- Fixes renumbering of new entities in Cartesian meshes after 2D de-refinement
  (\pr{1055}).
- Fixes various issues in the management of the
  \arcanemat{AllCellToAllEnvCellAccessor} class (\pr{1071}, \pr{1133},
  \pr{1188})
- Ensures that the associated global variable is properly allocated when
  constructing a material variable (\pr{1056})
- Fixes incorrect return value of `ItemInternal.Cell()` in the C# wrapper
  (\pr{1131})
- Fixes version 2 serialization when using `MPI_ANY_SOURCE` (\pr{1116})
- Fixes missing synchronization update when de-refining but not refining
  afterward (\pr{1266})
- Fixes various compilation issues (\pr{1044}, \pr{1047}, \pr{1280}, \pr{1309})
- Fixes potential incorrect SIMD padding update in the update of
  \arcane{ItemGroup} (\pr{1165})
- Uses the CMake variable `CMAKE_CUDA_HOST_COMPILER` to specify the host
  compiler associated with CUDA instead of enforcing the NVIDIA compiler option
  `-ccbin` (\pr{1386})
- Fixes compilation with ROCM 6.0 (\pr{1170})
- Fixes compilation with Swig 4.2 (\pr{1098})
- Fixes compilation with LibXml2 versions `2.12.0` and later (\pr{1019}).

### Internal

- Uses a reference counter for `RunQueueImpl` (\pr{996})
- Adds experimental support for changing data values from the command line
  (\pr{1038}).
- Adds a new hash calculation mechanism for \arcane{IData} implementations
  (\pr{1036}).
- Uses a new Hash mechanism for detecting variable value validity during
  recovery (\pr{1037}).
- Simplifies the \arcane{MDSpan} implementation using C++20 (\pr{1027})
- Moves Aleph implementation files into the source code instead of having them
  in header files and fixes the build system when using Trilinos (\pr{1002},
  \pr{1003})
- Adds classes to specify additional arguments for the methods
  \arcane{IMeshModifier::addCells()} and \arcane{IMeshModifier::addFaces()}
  (\pr{1077})
- Removes the old Cartesian mesh generator implementation (\pr{1069})
- Uses \arcane{Int64x2}, \arcane{Int64x3}, \arcane{Int32x2}, and
  \arcane{Int32x3} instead of `std::array` to manage Cartesian mesh build
  information (\pr{1067})
- Removes coverity warnings (\pr{1060}, \pr{1065}, \pr{1333}, \pr{1336},
  \pr{1378})
- Refactoring of the internal mechanism for managing material entity information
  to make properties accessible on the accelerator (\pr{1057}, \pr{1058},
  \pr{1086})
- Adds support for JSON format for \arcane{VariableDataInfo} (\pr{1148})
- Uses \arccore{ReferenceCounterImpl} for the implementation of
  \arccore{IThreadImplementation} and \arcane{IParallelDispatchT} (\pr{1132},
  \pr{1127})
- Fixes various compilation warnings (\pr{1130}, \pr{1163}, \pr{1164},
  \pr{1042}, \pr{1118}, \pr{1346})
- Improves Windows porting (\pr{1154}, \pr{1157}, \pr{1246})
- Adds the method \arcane{IMesh::computeSynchronizeInfos()} to force
  recalculation of synchronization information (
  \pr{1124})
- Disables floating-point exceptions at the end of calculation. This prevents
  floating-point exceptions (FPE) due to speculative executions (\pr{1232})
- Adds experimental class \arcane{DualUniqueArray} to manage a dual view on CPU
  and accelerator (\pr{1215})
- Uses the correct page size value for the internal allocator linked to CUDA
  (\pr{1198})
- Adds detection of MPI GPU Aware mode for ROCM (\pr{1209})
- Adds the method \arcane{ITimeStats::resetStats()} to reset the temporal
  statistics of the instance (\pr{1122}, \pr{1128})
- Adds implementation of \arccore{IThreadImplementation} and
  \arcane{AtomicInt32} using the standard C++ library (
  \pr{1339}, \pr{1340}, \pr{1342}, \pr{1343})
- Fixes possible duplication of uniqueId() in the boundary merging test
  (\pr{1362})

### Continuous Integration (CI)

- Updates the `U22_G12_C15_CU122` image (\pr{1031})
- Removes the use of targets using CUDA 11.8 ( \pr{1078})
- Updates the `vcpkg` workflow to version `2023-12-12` (\pr{1076})
- Uses C++20 for the `compile-all-vcpkg` workflow (\pr{1048})
- By default, sets `CMAKE_BUILD_TYPE` to `Release` if this variable is not
  specified during configuration (\pr{1149})
- Adds the macro `ARCANE_HAS_ACCELERATOR` if %Arcane is compiled with
  accelerator support (\pr{1123})
- Adds support for using 'googletest' with MPI cases (\pr{1092})
- Removes the use of Clang13 in CI (\pr{1083})
- Modifies the IFPEN workflow to compile all `framework` components directly
  (\pr{1214}, \pr{1281})
- Deactivates ongoing tests if a modification is published in the 'pull-request'
  branch (\pr{1250})
- Adds workflow for IFPEN 2021 with RHEL versions 7, 8, and 9 (\pr{1255})
- Updates certain obsolete actions (\pr{1218},\pr{1222})
- Uses C++20 for certain workflows (\pr{1189})

### Arccore

- Adds support for specific destructors in \arccore{ReferenceCounterImpl}
  (\pr{1068})
- Adds a new implementation of \arccore{ReferenceCounterImpl} that does not
  directly call the `operator delete`. This allows the instance to be destroyed
  externally (\pr{989}, \pr{1080}, \pr{1081}, \pr{1120}, \pr{1161}, \pr{1162}).
- Allows the use of \arccore{ReferenceCounterImpl} without needing an interface
  class (\pr{1121})
- Uses `std::atomic_flag` instead of `glib` for the \arccore{SpinLock} class
  (\pr{1110})
- Starts support for \arccore{Float16}, \arccore{Float32}, and
  \arccore{BFloat16} types (\pr{1087}, \pr{1088}, \pr{1089}, \pr{1095},
  \pr{1099}, \pr{1101}, \pr{1102}, \pr{1106})
- Makes the `StringImpl` class private (\pr{1096}, \pr{1097})
- Various improvements in \arccore{ISerializer} (\pr{1090}, \pr{1093},
  \pr{1112})
- Makes the method \arccore{Array::resizeNoInit()} public (\pr{1220}, \pr{1297})
- Adds a template argument in \arccore{Span}, \arccore{SmallSpan}, and
  \arccore{SpanImpl} to indicate the minimum index value (`0` by default)
  (\pr{1296}).
- Adds implicit and explicit conversions between \arccore{Array} and
  \arccore{SmallSpan} (\pr{1277}, \pr{1279})
- Fixes multiple conversions between UTF-8 and UTF-16 for \arccore{String}
  (\pr{1251}).
- Adds a utility function to fill an \arccore{Span} with random values
  (\pr{1103})
- Adds the method \arccore{arccoreCheckRange()} to check if a value is within a
  range (\pr{1091}).
- Adds \arccore{BuiltInDataTypeContainer} class to generalize operations on
  basic data types \pr{1105}
- Adds the possibility of retrieving the MPI communicator in
  \arccore{MessagePassing::IMessagePassingMng} (\pr{1248})
- Starts internal re-organization to make the use of 'Glib' optional
  (\pr{1328}).

### Axlstar

- Improves documentation management (\pr{1052}, \pr{1070}, \pr{1072}, \pr{1073},
  \pr{1084}, \pr{1104}, \pr{1107})
- Adds option to change the installation directory (\pr{1171})
- Adds command-line option to change the generated namespace name (\pr{1321}).
- Adds namespace name to the header file protection (\pr{1322}).

### Alien

- Starts support for filling matrices on the accelerator (\pr{1177}, \pr{1216},
  \pr{1242}, \pr{1245}, \pr{1252}, \pr{1267})
- Improves Alien documentation and manages it simultaneously with Arcane
  documentation (\pr{1265})
- Adds block management for the PETSc backend (\pr{1286}, \pr{1302})
- Adds the possibility of passing additional parameters in the Alien C API
  (\pr{1259})

## Arcane Version 3.11.15 (November 23, 2023) {#arcanedoc_version3110}

### New Features/Improvements

- Internal reorganization of synchronizations to use a single memory buffer for
  all synchronizations of the same mesh
  (\pr{861}, \pr{862}, \pr{863}, \pr{866}, \pr{867}, \pr{871}, \pr{872},
  \pr{873}, \pr{874}, \pr{878}, \pr{880})
- Adds interface \arcane{IVariableSynchronizerMng} to manage all instances of
  \arcane{IVariableSynchronizer} for a given mesh (\pr{869}, \pr{879}).
- Adds automatic mode to check if a synchronization had an effect. This mode is
  activated if the environment variable
  `ARCANE_AUTO_COMPARE_SYNCHRONIZE` is set. When this mode is active, statistics
  at the end of the calculation allow knowing the number of synchronizations for
  which ghost meshes were modified. The page \ref
  arcanedoc_debug_perf_compare_synchronization indicates how to use this
  mechanism (\pr{897}, \pr{898}, \pr{900}, \pr{902}, \pr{910}, \pr{926}).
- Adds two experimental classes \arcane{CartesianMeshCoarsening} and
  \arcane{CartesianMeshCoarsening2} to coarsen the initial Cartesian mesh. This
  currently works only in 2D
  (\pr{912}, \pr{913}, \pr{917}, \pr{918}, \pr{937}, \pr{942}, \pr{944},
  \pr{945}).
- Adds iterator (\arcane{ICartesianMesh::patches()}) over the patches of
  Cartesian meshes (\pr{948}).
- Adds class \arcane{CartesianPatch} to encapsulate the
  \arcane{ICartesianMeshPatch} (\pr{971}).
- Adds cumulative values for message exchange statistics across all executions
  (\pr{852}, \pr{853}).
- Adds C# support for data set functions (\pr{797}, \pr{800}, \pr{801},
  \pr{803}, \pr{804}).
- Adds support for ILU and IC preconditioners in parallel in the Aleph PETSc
  component (\pr{789}, \pr{799}).

#### Accelerator API

- In \arcaneacc{Reducer}, uses `HostPinned` memory instead of standard host
  memory to retain the reduced value. This allows for faster memory copying
  between GPUs
  (\pr{782})
- Adds missing ARCCORE_HOST_DEVICE macros for array size checking methods
  (\pr{785}).
- Adds method \arcaneacc{RunQueue::platformStream()} to retrieve a pointer to
  the associated native instance (cudaStream or hipStream, for example)
  (\pr{796}).
- Adds accelerator support for version 6 and version 8 of material
  synchronizations (\pr{855}).
- Adds support to retrieve the number of media in a mesh (\pr{860})
- Adds views on the environment variable (\pr{904}).
- Adds support for inclusive and exclusive Scan algorithms via the
  \arcaneacc{Scanner} class (\pr{921}, \pr{923}).
- Adds accelerator access to certain methods of \arcanemat{AllEnvCell} and
  \arcanemat{EnvCell} (\pr{925}).
- Adds support for array filtering (\pr{954}, \pr{955}).
- Adds asynchronous copy method for \arcane{NumArray} and variables on the
  mesh (\pr{961}, \pr{962})
- Adds asynchronous fill method \arcane{NumArrayBase::fill()}
  (\pr{963}, \pr{964}).

### Changes

- Changes the return type of
  \arcanemat{IMeshMaterialMng::synchronizeMaterialsInCells()} to return a
  `bool` (instead of a `void`) if materials were modified (\pr{827}).

### Corrections

- In the MSH format, corrects incorrect reading of node groups when there is
  only one element in that group (\pr{784}).
- Corrects compilation of \arcane{ArrayExtentsValue} for dimension 2 with Clang
  and NVCC compilers (\pr{786}).
- Corrects compilation of C# examples if the environment is not available. Also
  sets the environment variable
  `LD_LIBRARY_PATH` if necessary (\pr{811}).
- Corrects the optimization mode
  \arcanemat{eModificationFlags::OptimizeMultiMaterialPerEnvironment} to behave
  the same as other value update optimization modes when transitioning from a
  partial mesh to a pure mesh. The expected behavior is to take the partial
  material value and the partial media value. To maintain backward
  compatibility, this mode is not active by default. The method
  \arcanemat{IMeshMaterialMng::setUseMaterialValueWhenRemovingPartialValue()}
  allows it to be activated (\pr{844}, \pr{957}).
- Corrects a bug in \arcane{DoF} synchronization under certain conditions
  (\pr{920}).
- Corrects memory leak in the accelerator management of \arcanemat{AllEnvCell}
  (\pr{931}).
- Corrects incorrect accounting of additional meshes for complex options with
  multiple occurrences (\pr{941}).

### Internal

- Starting support for Cartesian meshes with structured block patches
  (\pr{946}).
- Starting refactoring of material connectivity management. The goal of this
  refactoring is to optimize this connectivity management to avoid having to
  call \arcanemat{IMeshMaterialMng::forceRecompute()} after a modification.
  These mechanisms are not active by default
  (\pr{783}, \pr{787}, \pr{792}, \pr{794}, \pr{795}, \pr{825}, \pr{826},
  \pr{828}, \pr{829}, \pr{831}, \pr{832}, \pr{835}, \pr{836}, \pr{838},
  \pr{839}, \pr{840}, \pr{841}, \pr{842}, \pr{843}, \pr{847}).
- Cleanup and internal reorganizations (\pr{781}, \pr{810}, \pr{813}, \pr{822},
  \pr{830}, \pr{834}, \pr{846}, \pr{856}, \pr{857}, \pr{868}, \pr{908},
  \pr{914}, \pr{952})
- Adds a test service to connect to a Redis database for the
  \arcane{BasicReaderWriter} service (\pr{780})
- Adds JSON format saving of metadata for protections/restorations. This format
  is not used yet but will eventually replace the XML format (\pr{779},
  \pr{865}).
- Created two classes \arcane{VariableIOReaderMng} and
  \arcane{VariableIOWriterMng} to manage the input/output part of
  \arcane{VariableMng} (\pr{777}).
- Adds method \arcane{JSON::value()} to return an \arccore{String} (\pr{778})
- Adds a new test module for materials
  (MaterialHeatModule). This module allows better testing of material mesh
  addition and removal methods (\pr{788}, \pr{790}, \pr{824}, \pr{848}).
- Improves the usage of \arcane{IProfilingService} (\pr{791})
- Integrates Alien sources into the same repository as %Arcane (the
  *framework* repository) (\pr{798}, \pr{812}, \pr{816}, \pr{817}, \pr{819},
  \pr{820}, \pr{883}, \pr{890}, \pr{891}, \pr{892}).
- Integrates the *Neo* mesh manager into the repository (\pr{802}, \pr{805},
  \pr{807}, \pr{808}, \pr{814}, \pr{815}, \pr{854}, \pr{881}, \pr{882},
  \pr{888}).
- Separates the \arcane{IIncrementalItemConnectivity} interface into two
  interfaces, \arcane{IIncrementalItemSourceConnectivity} and
  \arcane{IIncrementalItemTargetConnectivity}, to allow for connectivities that
  do not have targets (\pr{846}).
- Adds tests for version 3 of \arcane{FaceUniqueidBuilder} (\pr{850})
- Starts MacOS porting (\pr{884}, \pr{885})
- Optimizes EnsightGold output when the number of mesh types is very large
  (polyhedral meshes) (\pr{911}).
- Adds internal API to %Arcane for \arcane{ICartesianMesh} (\pr{943}).
- Creates the \arcane{MeshHandle} for additional meshes before calling the
  `Build` entry points (\pr{947}).
- Uses `std::atomic_ref` instead of `std::atomic` for accelerator reduction
  management (\pr{1352})

### Continuous Integration (CI) and Compilation

- Adds example compilation in CI (\pr{809}).
- Updates docker images in CI (\pr{818})
- For Arccore, compiles with the *Z7* option under Windows and adds debug
  symbols. The *Z7* option allows the use of the *ccache* tool (\pr{821})
- Uses 'GitHub actions' to simplify CI. This allows for composable high-level
  actions rather than duplicating code in every workflow (\pr{849}).
- Uses a docker image for CI with 'Circle-CI' and adds an execution for ARM64
  platforms (\pr{887}).
- Adds CMake variable `ARCANEFRAMEWORK_BUILD_COMPONENTS` allowing specification
  of components to compile. This variable is a list that can contain `%Arcane`
  and `Alien`. By default, both components are compiled (\pr{877})
- Deletes artifacts after retrieval to save disk space (\pr{915}).
- Removes default activation of the CMake variable
  `ARCANE_ADD_RPATH_TO_LIBS`. This variable is obsolete and will be removed
  later (\pr{919}).
- Removes the test output directory in CI to save disk space (\pr{924}).

### Arccore

- Internal cleanup of statistics management for message exchange (\pr{851})
- Corrects a bug in the constructors of \arccore{SharedArray2} from
  \arccore{UniqueArray2} or \arccore{Span} (\pr{899}).

### Axlstar

- Improves support for old versions of Doxygen (\pr{823}).

___

## Arcane Version 3.10.11 (June 30, 2023) {#arcanedoc_version3100}

Starting with version 3.10, internal connectivity management modifications are
planned to reduce memory consumption for Cartesian or structured meshes. The
page \ref arcanedoc_new_optimisations_connectivity describes the planned
evolutions and potential modifications required in %Arcane user codes.

### Deprecated Elements

- Removes methods that have been obsolete for several years from materials
  (\pr{652}).
- Deprecates the method \arcane{ItemConnectedListView::localIds()} which allows
  access to the localId() arrays of entities connected to another (\pr{666}).
- Makes internal %Arcane methods of \arcane{ItemInternalConnectivityList} and
  \arcane{ItemInternal} private or obsolete (\pr{787}).
- Deprecates \arcane{IDeflateService}. Use \arcane{IDataCompressor} instead
  (\pr{706}).
- Deprecates \arcane{IPostProcessorWriter::setMesh()} which does nothing by
  default. The desired mesh must be specified when building the service (via
  \arcane{ServiceBuilder}) (\pr{748}).
- Deprecates \arcane{IHashAlgorithm::computeHash()}. Use the
  \arcane{IHashAlgorithm::computeHash64()} version instead. Adds the methods
  \arcane{IHashAlgorithm::hashSize()} and \arcane{IHashAlgorithm::name()} to
  retrieve algorithm information and allow dynamic creation via a service
  (\pr{696}, \pr{707}).
- Deprecates the methods \arccore{ArrayView::range()}, \arccore{Span::range()}
  and \arccore{AbstractArray::range()}. These methods generate temporary
  objects, which can cause problems when used in `for-range` loops (see
  'Temporary range expression'
  in [range-for](https://en.cppreference.com/w/cpp/language/range-for)). You can
  directly use the `begin()` or `end()` methods instead (\pr{757}).

### New Features/Improvements

- Creation of a \arcane{SmallArray} class to manage small-sized arrays with
  stack-allocated memory (\pr{615}, \pr{732}).
- Adds the possibility in the Aleph PETSc implementation to pass arguments that
  will be used to initialize PETSc via the call to `PetscInitialize()`
  (\pr{621}).
- Adds a post-processing writer in `VTK HDF V2` format. This format allows
  multiple post-processing runs to be placed in the same HDF5 file
  (\pr{637}, \pr{638}, \pr{639}).
- Improves memory management of connectivities during mesh creation. Arrays are
  pre-allocated to avoid successive copies when adding entities one by one
  (\pr{689}, \pr{763}).
- Adds method \arcane{MDSpan::slice()} to return a view on a sub-part of the
  initial view (\pr{690}).
- Possibility to dynamically calculate the output directory name in
  \arcane{SimpleTableWriterHelper} (\pr{607}).
- Adds SHA3 hash calculation functions (\pr{695}, \pr{697}, \pr{705}).
- Adds \arcane{ItemGenericInfoListView} class to make generic entity information
  accessible on the accelerator (such as \arcane{Item::owner()},
  \arcane{Item::uniqueId()}, ...) (\pr{727}).
- Optimizes \arcane{ItemGroup::applyOperation()} to avoid going through
  sub-groups and to directly use the base group if all entities are of the same
  type. This mechanism is not active by default. The environment variable
  `ARCANE_APPLYOPERATION_VERSION` must be set to `2` to use it (\pr{728}).
- Adds operators `-=`, `*=` and `/=` for views (via \arcane{DataSetter})
  {\pr{733}).
- Adds \arcane{Vector3} class to generalize \arcane{Real3} for other types
  (\pr{750}).
- Adds events for \arcane{IMesh} to be notified of calls to
  \arcane{IMesh::prepareForDump()} \pr{771}.

#### Accelerator API

- Support for memory copies with indexing on the accelerator (\pr{617},
  \pr{625}, \pr{658}, \pr{773})
- Partial integration of CUPTI (CUDA Profiling Tools Interface)
  allowing automatic retrieval of profiling information on NVIDIA cards. For
  example, this allows retrieving information on memory transfers between the
  CPU and the GPU (\pr{627}, \pr{632}, \pr{642}).
- CUDA support for tracing managed memory allocations/deallocations and for
  allocating multiple page-sized blocks (\pr{641}, \pr{672}, \pr{685},
  \pr{693}).
- Support for synchronizations with MPI's 'Accelerator Aware' mode. This allows
  synchronizations without memory copying between the CPU and the accelerator.
  This mechanism is also available in shared memory message exchange mode and
  hybrid mode (\pr{631}, \pr{644}, \pr{645}, \pr{646}, \pr{654}, \pr{661},
  \pr{680}, \pr{681}, \pr{765}).
- Accelerator support to determine if a given memory address is accessible on
  the accelerator, on the CPU, or both. Also adds two macros
  ARCANE_CHECK_ACCESSIBLE_POINTER() and ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS()
  to verify that a memory region is to be used in a \arcaneacc{RunQueue}
  (\pr{660}).
- Accelerator support to specify information about variable memory allocation
  (\arcane{IVariable::setAllocationInfo()}). This allows, for example,
  indicating whether a variable will be accessed on the accelerator or on the
  CPU (\pr{684}).
- Adds method \arcane{MeshUtils::markMeshConnectivitiesAsMostlyReadOnly()} to
  indicate that the variables managing connectivity will not be frequently
  modified. This allows optimizing memory management between the accelerator and
  CPU to avoid copies. By default, entity groups (\arcane{ItemGroup}) use this
  attribute (\pr{691}, \pr{714}).
- Makes information stored in \arcanemat{AllEnvCell} accessible on the
  accelerator (\pr{742}).
- Adds views based on \arccore{Span} to the accelerator API. This allows having
  views on containers other than those of %Arcane (for example `std::vector`) as
  soon as the data is contiguous in memory (\pr{770}).
- Allows copying \arcane{NumArray} from different memory regions (\pr{651}).
- Support for the new interface \arccore{IMemoryAllocator3} in accelerator
  allocators (\pr{671}, \pr{674}).

### Changes

- Allows copying an instance of \arcaneacc{Runner} using reference semantics
  (\pr{623}).
- On the accelerator, a specific kernel is used by default across the entire
  grid for reductions. Previously, a kernel was used that mixed atomic
  operations and calculation on blocks (\pr{640}).
- Uses accelerator memory for reduction operations. Previously, managed memory
  was used (\pr{643}, \pr{683}).
- Changes the node numbering of hexagonal meshes for `HoneyCombMeshGenerator`
  (\pr{657}).
- Groups the functions of \arcane{mesh_utils} and \arcane{meshvisitor} into the
  \arcane{MeshUtils} namespace (\pr{725}).
- Moves internal %Arcane methods from \arcane{IMesh} and \arcane{IItemFamily}
  into an internal interface (\pr{726}, \pr{738}, \pr{752}, \pr{768}).
- In bit-by-bit comparisons, differences are not considered if both values being
  compared are `NaN`.
- Adds display of time spent in MPI initialization and accelerator runtime
  (\pr{760}).
- Automatically calls \arcane{ICartesianMesh::computeDirections()} after a call
  to \arcane{IMesh::prepareForDump()}. This ensures that Cartesian information
  is consistent after potential compaction (\pr{772}).

### Corrections

- Correctly handles the destruction of \arcane{StandaloneSubDomain} singleton
  instances. Previously, the instance was destroyed in global destructors after
  `main()` finished, which could cause problems in some cases (\pr{619}).
- Corrects errors in the copy constructor of \arcane{NumArray} (\pr{717}).
- In \arcane{FloatingPointExceptionSentry}, the exception handling flag is set
  unconditionally. Previously, it was tested whether exceptions were active, and
  if not, nothing was done. Since this detection mechanism is not always
  reliable, it is removed (\pr{720}).
- Saves the mesh modification `timestamp` (\arcane{IMesh::timestamp()}) and the
  `need-compact` attribute in protections to ensure the same behavior with or
  without restoration. Notably,
  `need-compact` was always set to `true` during restoration, causing compaction
  to always occur at least once after restoration. Since the entities were
  compacted, this did not change the results, but it could cause reallocations
  that invalidated calculated structures such as Cartesian mesh information
  (\pr{739}, \pr{756})
- Corrects the use of \arcane{MeshReaderMng::setUseMeshUnit()} which was not
  taken into account when the data set language is French (\pr{754}).
- Removes unnecessary reallocation in \arccore{AbstractArray} when the new
  capacity is identical to the old one (commit cac7fae3c471f6).

### Internal

- Starting support for dynamically creating services containing data set
  options (\pr{613}).
- Removes use of \arcane{ISubDomain} in certain parts (\pr{620}).
- Adds function to retrieve command line arguments (\pr{624}).
- Removes coverity warnings (\pr{626}, \pr{692}).
- Makes certain methods of \arcane{Materials::IMeshComponent} `const`
  (\pr{630}).
- Various improvements to accelerator management (\pr{647}).
- Corrects compilation with PAPI 7.0 and PETSc 3.19 (\pr{648}).
- Adds a field of type \arcane{Int32} in different classes managing
  connectivities to handle an offset on the localId(). Currently, this is not
  used and the offset is always 0 (\pr{649}, \pr{712}, \pr{723}, \pr{736},
  \pr{737}, \pr{744})
- Support for using a specific parallel driver to run tests (\pr{663}).
- Replaces the use of `ENUMERATE_*` to access connectivities with for-range
  (\pr{666}, \pr{759}).
- Adds a specific interface to create Cartesian meshes. This will eventually
  allow specialized methods for these meshes to be provided so that generation
  is faster and directly uses the correct numbering (\pr{694}, \pr{749},
  \pr{751}).
- Adds typedefs in \arcane{MDSpan} to retrieve the type of the element and the
  Layout (\pr{699}).
- Adds support for using common hashes in \arcane{BasicReaderWriter}, which can
  be used for bit-by-bit comparisons and generalizes the mechanism for accessing
  this hash base (\pr{698}, \pr{700}, \pr{701}).
- Adds adapter for the Redis database {\pr{702}).
- Internal refactoring of the synchronization mechanism to make it independent
  of the data type (\pr{704}, \pr{708}, \pr{709}, \pr{711})
- Uses a single buffer for multiple variable synchronization instead of going
  through serialization (\pr{710}).
- Adds \arcane{MeshKind} class to manage properties on the mesh structure
  (Cartesian, unstructured, AMR, ...) (\pr{718}).
- Adds a specific macro for obsolete methods, which will not be immediately
  removed, in order to disable compilation warnings for these methods. This
  allows user code to suppress compilation warnings for these methods if the
  macro `ARCANE_NO_DEPRECATED_LONG_TERM` is defined during compilation
  (\pr{722}).
- Adds the possibility to display the CPU affinity of all ranks (\pr{729}).
- For `VTK HDF` formats, adds information on the \arcane{Item::uniqueId()} of
  nodes and meshes (\pr{741}).
- Improves continuous integration to avoid running tests if only certain files
  are modified (for example, only `.md`) and adds date and license verification
  (\pr{743}, \pr{745}).
- Makes the methods of \arcane{ItemInternalConnectivityList} internal to %Arcane
  and simplifies class management by grouping connectivity information into a
  substructure (\pr{640}).
- Moves the \arcane{ItemGroupImplPrivate} class into its own file (\pr{730}).
- Makes the function Arcane::arcaneCheckAt() `constexpr` (\pr{746}).

### Arccore (2.5.0)

- Propagates the source allocator in the constructor and copy constructor of
  \arccore{UniqueArray} and \arccore{UniqueArray2} (\pr{635}, \pr{656}).
- Uses \arccore{Span} instead of \arccore{ConstArrayView} for certain arguments
  to allow views whose size exceeds 2Go (\pr{635}).
- Avoids the construction of default elements that will subsequently be
  overwritten in \arccore{AbstractArray::_resizeAndCopyView()}. This also allows
  this method to be used with data types that do not have a default constructor
  (\pr{635}).
- No longer performs minimal allocation even if an allocator other than the
  default allocator is used. Previously, at least 4 elements were always
  allocated (\pr{635}).
- Corrects unnecessary double allocation in \arccore{Array::operator=()} if the
  two arrays do not have the same allocator (\pr{655}).
- Allows displaying the exception message in the \arccore{Exception}
  constructor. This is useful for debugging exceptions outside of
  `try{ ... } catch` or exceptions that throw other exceptions (\pr{659}).
- Adds \arccore{IMemoryAllocator3} interface which enriches
  \arccore{IMemoryAllocator} to pass more information to the allocator. This
  allows, for example, adding the allocated size or the array name (\pr{662},
  \pr{673}, \pr{677}, \pr{713}, \pr{719}).
- Adds `Int8` and `BFloat16` types to Arccore::eBasicDataType (\pr{669})
- Adds various conversion functions between \arccore{Span} and `std::array`.
  Also adds the `subPart` and `subPartInterval` methods common to
  \arccore{ArrayView}, \arccore{ConstArrayView} and \arccore{Span} (\pr{670}).
- Removes coverity warnings (\pr{675}).
- Support for naming \arccore{Array} arrays. This is used in
  \arccore{IMemoryAllocator3} to display allocation information (\pr{676},
  \pr{682}).
- Moves operators such as '==', '!=', '<<' and '<' into the corresponding
  classes as a `friend` function (\pr{703}).
- Deprecates the methods \arccore{ArrayView::range()}, \arccore{Span::range()}
  and \arccore{AbstractArray::range()}. These methods generate temporary
  objects, which can cause problems when used in `for-range` loops. You can
  directly use the `begin()` or `end()` methods instead (\pr{757}).

### Axlstar (2.2.0)

- Adds, for experimentation, the possibility of specifying multiple types
  (`caseoption`, `subdomain`, ...) for services (\pr{715}).

___

## Arcane Version 3.9.5 (April 4, 2023) {#arcanedoc_version390}

### New Features/Improvements

- Adds Arcane::geometric::GeomElementViewBase::setValue() method to modify the
  value of a coordinate (\pr{598}).
- Optimizes the search for values in lookup tables by using dichotomy
  (\pr{596}).
- Accelerator API support for media via the macro RUNCOMMAND_MAT_ENUMERATE()
  (\pr{595},\pr{593}, \pr{588}, \pr{586}, \pr{577}).
- Adds explicit constructors between Arcane::Real2 and Arcane::Real3 (\pr{591}).

### Changes

- Adds information about the dimension of an entity in Arcane::ItemTypeInfo and
  verifies the consistency between the mesh dimension and the Arcane::Cell
  entities used in this mesh (\pr{567}).
- Adds converters from Arcane::ItemEnumeratorT to Arcane::ItemLocalIdT
  (\pr{564}).

### Fixes

- Fixes data set group retrieval in the case of multiple meshes. It always took
  the default mesh to search for groups even if the option was associated with
  another mesh (\pr{604}).
- Fixes incorrect search bar dimension in the documentation in certain cases
  (\pr{597}).
- Fixes incorrect usage of the initial partitioner when multiple meshes are
  present in the data set. The partitioner only defined variables on the first
  mesh, which introduced inconsistencies (\pr{592}).
- Fixes incorrect detection of Cartesian connectivities in the Y direction in
  3D (\pr{590}).
- Fixes parallel blocking in the VTK reader if there are only connectivities in
  the mesh (\pr{589}).
- Fixes incorrect mesh type when using sub-meshes ranging from a 2D mesh to a 1D
  mesh (\pr{587}).
- Fixes possible ambiguity when constructing classes derived from Arcane::Item
  (\pr{579}).

### Internal

- Uses a reference counter for Arcane::ICaseMng (\pr{603}).
- Preliminary support for creating an autonomous sub-domain (\pr{599}).
- Refactoring of buffer management during variable synchronizations to prepare
  for accelerator support (\pr{585}, \pr{582}, \pr{575}, \pr{572}, \pr{571},
  \pr{570}, \pr{569}, \pr{566}).
- Makes the constructors of Arcane::Materials::ComponentItemVectorView private
  (\pr{580}).
- Various modifications in Arcane::ConstMemoryView and Arcane::MutableMemoryView
  (\pr{574}, \pr{573}, \pr{562}).
- Adds a preliminary interface specific to a mesh for mesh allocation
  (\pr{568}).

### Arccore (version 2.2.0)

- Adds generic support via the Arccore::MessagePassing::GatherMessageInfo class
  for all types of `MPI_Gather` (\pr{556}).
- Adds distinction in Arccore::MessagePassing::MessageRank between
  `MPI_ANY_SOURCE` and `MPI_PROC_NULL` (\pr{555}).

___

## Arcane Version 3.8.15 (February 22, 2023) {#arcanedoc_version380}

### New Features/Improvements

- Support for specifying default values of Arcane::ParallelLoopOptions on the
  command line in multi-thread mode (\pr{420}).
- Support for Lima files, MED files, and `msh` format files with mesh services
  (\pr{435}, \pr{439}, \pr{449}).
- Adds Arcane::NumArrayUtils::readFromText() function to fill an
  Arcane::NumArray instance from an ASCII file (\pr{444}).
- Support for reading parallel data from MED format files (\pr{449}).
- Support for reading node groups (Arcane::NodeGroup) in MSH format meshes
  (\pr{475}).
- Support for renumbering 3D AMR meshes. This allows the same 3D numbering
  regardless of the decomposition (\pr{495}, \pr{514}, \pr{523}).
- Adds access to Arcane::IMeshMng in Arcane::ICaseMng and
  Arcane::IPhysicalUnitSystem (\pr{461}).
- Accelerator support for classes managing Cartesian meshes
  (Arcane::CellDirectionMng, Arcane::FaceDirectionMng, and
  Arcane::NodeDirectionMng) (\pr{474})
- Adds Arcane::impl::MutableItemBase class to replace the use of
  Arcane::ItemInternal (\pr{499}).
- Adds the possibility to index components of Arcane::Real2, Arcane::Real3,
  Arcane::Real2x2, and Arcane::Real3x3 using the `operator()` (\pr{485}).
- Preliminary developments for multi-dimensional mesh variables (\pr{459},
  \pr{463}, \pr{464}, \pr{466}, \pr{471}).
- Adds Arcane::IDoFFamily interface to manage Arcane::DoF. Previously, the
  Arcane::mesh::DoFFamily implementation had to be used directly (\pr{480})
- Support in Aleph for variables that do not have default families (such as
  Arcane::DoF) (\pr{468}).
- Support for compressing Arcane::IData data via an instance of
  Arcane::IDataCompressor. This mechanism is available for materials by calling
  the method
  Arcane::Materials::IMeshMaterialMng::setDataCompressorServiceName(). It is
  used when calling Arcane::Materials::IMeshMaterialMng::forceRecompute() or
  using the Arcane::Materials::MeshMaterialBackup class (\pr{531}, \pr{532}).
- Support for multiple meshes in data set options (\pr{453}, \pr{548}).
- Adds Arcane::MeshHandleOrMesh class to facilitate the transition between
  Arcane::IMesh and Arcane::MeshHandle during initialization (\pr{549}).

### Changes

- Always uses a class instead of an integer to specify dimensions (extents) in
  the Arcane::NumArray and Arcane::MDSpan classes. This allows getting closer to
  the implementation planned in the C++23 standard and having static
  (compile-time known) dimensions (\pr{419}, \pr{425}, \pr{428}).
- Removes timers that use CPU time instead of elapsed time. The
  Arcane::Timer::TimerVirtual type still exists but now behaves like the
  Arcane::Timer::TimerReal type (\pr{421}).
- Removes the template parameter with the array rank in the
  Arcane::DefaultLayout, Arcane::RightLayout, and Arcane::LefLayout classes
  (\pr{436}).
- Deprecates methods in Arcane::ModuleBuildInfo that use Arcane::IMesh. Methods
  that use Arcane::MeshHandle must be used instead. (\pr{460}).
- Changes the return type of Arcane::IMeshBase::handle() so that it does not
  return a reference but a value (\pr{489}).
- Uses specific base classes by service type when generating `axl` files
  (\pr{472}).
- Uses a new class Arcane::ItemConnectedListView (instead of
  Arcane::ItemVectorView) to manage connectivities on entities. Methods such as
  Arcane::Cell::nodes() now return an object of this type. The purpose of this
  new class is to be able to propose a data structure specific to connectivities
  and another specific to entity lists (such as Arcane::ItemGroup groups).
  Conversion operators have been added to ensure compatibility with existing
  source code (\pr{534},\pr{535}, \pr{537}, \pr{539})
- New VTK HDF format dump service. This format is only supported by Paraview
  versions 5.11+. The current implementation is experimental (\pr{510},
  \pr{525}, \pr{527}, \pr{528} \pr{554}, \pr{546}).
- Moves header files of the `arcane_core` component, which were in the root of
  `arcane`, to a subdirectory
  `arcane/core`. To remain compatible with existing header files referencing
  these new files, they are generated during installation.
- Adds an `arcane_hdf5` component containing utility classes (such as
  Arcane::Hdf5::HFile, ...). The corresponding header files are now in the
  `arcane/hdf5` directory (\pr{505}).
- Cleanup of HDF5 managing classes: deprecates the copy constructor of classes
  in `Hdf5Utils.h` and removes support for HDF5 versions older than 1.10
  (\pr{526}).
- Allows the creation of null instances of Arcane::ItemGroup before the
  initialization of %Arcane. This allows, for example, having global
  Arcane::ItemGroup variables or derived classes (\pr{544}).
- Modifies the behavior of the ENUMERATE_COMPONENTITEM macro to use a type
  instead of a character string in the enumerator name. This allows it to be
  used with a template parameter (\pr{540}).

### Fixes

- Fixes possible inconsistency between connectivities stored in
  Arcane::ItemConnectivityList and Arcane::mesh::IncrementalItemConnectivity
  (\pr{478}).
- Fixes incorrect value of Arcane::HashTableMapT::count() after calling
  Arcane::HashTableMapT::clear() (\pr{506}).

### Internal

- Removes obsolete internal classes Arcane::IAllocator,
  Arcane::DefaultAllocator, Arcane::DataVector1D, Arcane::DataVectorCommond1D,
  and Arcane::Dictionary. These classes have not been used for a long time
  (\pr{422}).
- Adds Arcane::TestLogger class to compare test results against a reference
  listing file (\pr{418}).
- Adds the possibility to keep instances of Arcane::ItemSharedInfo' in
  Arcane::ItemInternalConnectivityList'. This will allow removing an indirection
  when accessing connectivities. This option is currently only used in testing
  (\pr{371})
- Adds support for calling Arccore::MessagePassing::mpLegacyProbe() for the
  different available message exchange modes (\pr{431})
- Refactoring of Arcane::NumArray, Arcane::MDSpan, Arcane::ArrayExtents, and
  Arcane::ArrayBounds classes to unify the code and support both static and
  dynamic dimensions. The page \ref arcanedoc_core_types_numarray explains the
  use of these classes (\pr{426}, \pr{428}, \pr{433}, \pr{437}, \pr{440}).
- By default, uses Version 2 of synchronizations with MPI. This version is the
  same as version 1 used previously but without support for derived types
  (\pr{434}).
- [accelerator] Unifies the launching of calculation kernels created by the
  macros RUNCOMMAND_LOOP and RUNCOMMAND_ENUMERATE (\pr{438}).
- Unifies the profiling API between commands (Arcane::Accelerator::RunCommand)
  and classic enumerators (via Arcane::IItemEnumeratorTracer). At the end of the
  calculation, the display is sorted by decreasing time spent in each loop
  (\pr{442}, \pr{443}).
- Starts development of the Arcane::NumVector and Arcane::NumMatrix classes to
  generalize the Arcane::Real2, Arcane::Real3, Arcane::Real2x2, and
  Arcane::Real3x3 types. These classes are currently for internal use of %Arcane
  (\pr{441}).
- Various optimizations in internal classes managing connectivities and
  iterators to reduce their size (\pr{479}, \pr{482}, \pr{483}, \pr{484})
- Removes the use of Arcane::ItemInternalList in Arcane::ItemVector and
  Arcane::ItemVectorView (\pr{486}, \pr{487}).
- Removes the use of Arcane::ItemInternalVectorView (\pr{498})
- Removes the use of Arcane::ItemInternal in many internal classes (\pr{488},
  \pr{492}, \pr{500}, \pr{501}, \pr{502})
- Removes additional indexers in Arcane::MDSpan and Arcane::NumArray for the
  Arcane::Real2, Arcane::Real3, Arcane::Real2x2, and Arcane::Real3x3 classes.
  These indexers were added for testing purposes but were not used (\pr{490}).
- Allows copying instances of Arcane::StandaloneAcceleratorMng and uses a
  reference semantics (\pr{509}).
- Allows instances of Arcane::NumVector and Arcane::NumMatrix with arbitrary
  values (previously only values 2 or 3 were allowed) (\pr{521})
- Moves the implementation of Aleph-related classes to source files instead of
  header files (\pr{504}).
- Provides an empty implementation for methods using Arcane::IMultiArray2Data.
  This interface is no longer used, and this will allow user code to remove
  visitors associated with this type (\pr{529}).

### Arccore

- Fixes bug if the method
  Arccore::MessagePassing::Mpi::MpiAdapter::waitSomeRequestsMPI() is called with
  already completed requests (\pr{423}).
- Adds a template parameter to Arccore::Span and Arccore::Span indicating the
  number of elements in the view. This allows managing views with a compile-time
  known number of elements, similar to `std::span` (\pr{424}).
- Adds Arccore::MessagePassing::mpLegacyProbe() functions whose semantics are
  similar to `MPI_Iprobe` and `MPI_Probe` (\pr{430}).
- Fixes empty request detection (\pr{427}, \pr{429}).
- Various improvements to the continuous integration mechanism (\pr{503},
  \pr{511}, \pr{512}, \pr{513})

### Axlstar

- Add support for using a specific mesh in service instance (\pr{451})
- Remove support to build with `mono` (\pr{465}).
- Remove support for 'DualNode' and 'Link' items (\pr{524}).
- Various improvements in documentation (\pr{530}).
- Add preliminary support for multi-dimension variables (\pr{520}).
- Fix: Add support of Doxygen commands in AXL descriptions (\pr{538})
- Fix: error with complex options containing more than 30 suboptions (\pr{533})

___

## Arcane Version 3.7.23 (November 17, 2022) {#arcanedoc_version370}

### New Features/Improvements:

- Complete refactoring of the documentation to make it more consistent and
  visually appealing (\pr{378}, \pr{380}, \pr{382}, \pr{384}, \pr{388},
  \pr{390}, \pr{393}, \pr{396})
- Adds a CSV format output management service (see
  \ref arcanedoc_services_modules_simplecsvoutput) (\pr{277}, \pr{362})
- Adds the possibility to specify the keyword `Auto` for the CMake variable
  `ARCANE_DEFAULT_PARTITIONER`. This allows automatically choosing the
  partitioner used during configuration based on those available (\pr{279}).
- Adds implementation of synchronizations that use the `MPI_Neighbor_alltoallv`
  function (\pr{281}).
- Reduction of memory footprint used for connectivity management following
  various internal changes.
- Optimizations during initialization (\pr{302}):
  - Uses `std::unordered_set` instead of `std::set` for checking uniqueId()
    duplication.
  - When creating a mesh, non-duplication of uniqueId() is only checked in check
    mode.
- Creates an Arcane::ItemInfoListView class to eventually replace
  Arcane::ItemInternalList and access entity information from their localId()
  (\pr{305}).
- [accelerator] Adds support for atomic Min/Max/Sum reductions for `Int32`,
  `Int64`, and `double` types (\pr{353}).
- [accelerator] Adds a new reduction algorithm without going through atomic
  operations. This algorithm is not used by default. It must be activated by
  calling Arcane::Accelerator::Runner::setDeviceReducePolicy() (\pr{365},
  \pr{379})
- [accelerator] Adds the possibility to change the number of threads per block
  when launching a command via
  Arcane::Accelerator::RunCommand::addNbThreadPerBlock() (\pr{374})
- [accelerator] Adds support for pre-loading (prefetching) advice for memory
  zones (\pr{381})
- [accelerator] Adds support for retrieving information about available
  accelerators and associating an accelerator with an instance of
  Arcane::Accelerator::Runner (\pr{399}).
- Start of development to be able to view an array variable on entities as a
  multi-dimensional variable (\pr{335}).
- Adds an Arcane::MeshHandle::onDestroyObservable() observable to be notified
  upon destruction of an Arcane::IMesh instance (\pr{336}).
- Adds the Arcane::mesh_utils::dumpSynchronizerTopologyJSON() method to save the
  communication topology for synchronizations in JSON format (\pr{360}).
- Adds the Arcane::ICartesianMesh::refinePatch3D() method to refine a 3D mesh
  into several AMR patches (\pr{386}).
- Adds implementation of reading hardware counters via the Linux perf API
  (\pr{391}).
- Adds support for automatically profiling commands launched via
  RUNCOMMAND_ENUMERATE (\pr{392}, \pr{394}, \pr{395})

### Changes:

- Modifies the classes associated with Arcane::NumArray (Arcane::MDSpan,
  Arcane::ArrayBounds, ...) so that the template parameter managing the rank is
  a class and not an integer. The ultimate goal is to have the same template
  parameters as the `std::mdspan` and `std::mdarray` classes planned for the C++
  2023 and 2026 standards. Dimensions must now be replaced by the keywords
  Arcane::MDDim1, Arcane::MDDim2, Arcane::MDDim3, or Arcane::MDDim4 (\pr{333})
- The Arcane::NumArray::resize() method no longer calls the default constructor
  for array elements. This was already the case for simple types (Arcane::Real,
  Arcane::Real3, ...) but now it is also the case for user types. This allows
  this method to be called when memory is allocated on the accelerator.
- Adds Arcane::ItemTypeId class to manage the entity type (\pr{294})
- The entity type is now stored as an Arcane::Int16 instead of an
  Arcane::Int32 (\pr{294})
- Removes obsolete methods from Arcane::ItemVector, `MathUtils.h`,
  Arcane::IApplication, Arcane::Properties, and Arcane::IItemFamily (\pr{304}).
- Refactoring of classes managing entity enumeration (\pr{308}, \pr{364},
  \pr{366}).
  - Removes the base class Arcane::ItemEnumerator from Arcane::ItemEnumeratorT.
    Inheritance is replaced by a conversion operator.
  - Simplifies Arcane::ItemVectorViewConstIterator
  - Simplifies the internal management of the `operator*` to avoid using
    Arcane::ItemInternal.
- Refactoring of the configuration file `ArcaneConfig.cmake` generated
  (\pr{318}):
  - No longer exports external packages by default in `ArcaneTargets.cmake`. The
    `ArcaneConfig.cmake` file now calls the CMake command `find_dependency`. The
    CMake variable `FRAMEWORK_NO_EXPORT_PACKAGES` is therefore no longer used by
    default.
  - Adds the variable `ARCANE_USE_CONFIGURATION_PATH` to `ArcaneConfig.cmake` to
    allow loading package paths from Arcane configuration. This variable is set
    to `TRUE` by default.
- Modifies the prototype of certain methods in classes implementing
  Arcane::IItemFamily to use Arcane::Item instead of Arcane::ItemInternal
  (\pr{311})
- Creates an Arcane::ItemFlags class to manage flags concerning object
  properties that were previously in Arcane::ItemInternal (\pr{312})
- Deprecates the `operator->` for the Arcane::Item class and derived classes
  (\pr{313})
- Changes the default value of face numbering in the Cartesian generation
  service to use Cartesian numbering (\pr{315})
- Modification of the signature of the methods in Arcane::IItemFamilyModifier
  and Arcane::mesh::OneMeshItemAdder to use Arcane::ItemTypeId instead of
  Arcane::ItemTypeInfo and Arcane::Item instead of Arcane::ItemInternal
  (\pr{322})
- Removes Arcane::Item::activeFaces() and Arcane::Item::activeEdges() methods
  which are no longer used (\pr{351}).
- [C#] Adds the possibility at the end of calculation to destroy instances of
  different managers, even when `.Net` support is not enabled. Previously, these
  managers were never destroyed to prevent potential crashes when the `.Net`
  environment's 'garbage collector' triggers. This destruction can be enabled by
  setting the environment variable `ARCANE_DOTNET_USE_LEGACY_DESTROY` to the
  value `0`. This is not active by default because there may still be issues
  with certain user services (\pr{337}).
- [configuration] It is now necessary to use at least CMake version 3.21 to
  compile or use #Arcane (\pr{367}).
- Adds a move constructor (`std::move`) for Arcane::NumArray (\pr{372}).
- [accelerator] Removes obsolete methods for creating
  Arcane::Accelerator::RunQueue and Arcane::Accelerator::Runner (\pr{397}).
- Deprecates the Arcane::AtomicInt32 class. The `std::atomic<Int32>` class must
  be used instead (\pr{408}).

### Fixes:

- Fixes bug when reading information with the `BasicReaderwriter` service when
  compression is active (\pr{299})
- Fixes bug introduced in version 3.6 that changed the output directory name for
  bit-by-bit comparisons with the
  `ArcaneCeaVerifier` service (\pr{300}).
- Fixes incorrect recalculation of the maximum number of entities connected to
  an entity in the case of particles (\pr{301})

### Internal:

- Simplifies the implementation of synchronizations providing a
  data-type-independent mechanism (\pr{282}).
- Uses variables to manage certain data on entities such as
  Arcane::Item::owner(), Arcane::Item::itemTypeId(). This will eventually allow
  this information to be accessed on accelerators (\pr{284}, \pr{285}, \pr{292},
  \pr{295})
- Adds an Arcane::ItemBase class serving as a base class for Arcane::Item and
  Arcane::ItemInternal (\pr{298}, \pr{363}).
- Removes an indirection when accessing connectivity information from an
  entity (for example Arcane::Cell::node()) (\pr{298}).
- Simplifies the management of common information for entities in a family so
  that there is now only one common instance of Arcane::ItemSharedInfo
  (\pr{290}, \pr{292}, \pr{297}).
- Removes certain uses of Arcane::ISubDomain (\pr{327})
  - Adds the possibility to create an Arcane::ServiceBuilder instance from an
    Arcane::MeshHandle.
  - Adds the possibility to create an Arcane::VariableBuildInfo instance via an
    Arcane::IVariableMng.
- Optimizes the structures managing the Cartesian mesh so that instances of
  Arcane::ItemInternal* no longer need to be stored. This reduces memory
  consumption and potentially improves performance (\pr{345}).
- Uses views instead of Arccore::SharedArray for classes managing Cartesian
  directions (Arcane::CellDirectionMng, Arcane::FaceDirectionMng, and
  Arcane::NodeDirectionMng) (\pr{347}).
- Uses a reference counter to manage Arccore::Ref<Arcane::ICaseFunction>
  (\pr{329}, \pr{356}).
- Adds a constructor for the Arcane::Item class and its derived classes from a
  localId() and an Arcane::ItemSharedInfo (\pr{357}).
- Updates C# project references to use the latest package versions (\pr{359}).
- Cleanup of Arcane::Real2, Arcane::Real3, Arcane::Real2x2, and Arcane::Real3x3
  classes and adds constructors from an Arcane::Real (\pr{370}, \pr{373}).
- Partial refactoring of concurrency management to pool certain
  functionalities (\pr{389}).
- Uses an Arccore::UniqueArray for the container of Arcane::ListImplT.
  Previously, the container was a simple C array (\pr{407}).
- In Arcane::ItemGroupImpl, uses Arcane::AutoRefT to keep references to
  sub-groups instead of a simple pointer. This ensures that the sub-groups will
  not be destroyed as long as the associated parent exists.
- Fixes various warnings reported by coverity (\pr{402}, \pr{403}, \pr{405},
  \pr{409}, \pr{410} )
- [C#] Indicates that at least language version 8.0 is required.

### Arccon:

Uses version 1.5.0:

- Add CMake functions to unify handling of packages arccon Arccon
  componentbuildBuild configuration (\pr{342}).

### Arccore:

Uses version 2.0.12.0:

- Remove some coverity warnings (\pr{400})
- Use a reference counter for IMessagePassingMng (\pr{400})
- Fix method asBytes() with non-const types (\pr{400})
- Add a method in AbstractArray to resize without initializing (\pr{400})
- Make class ThreadPrivateStorage deprecated (\pr{400})

___

## Arcane Version 3.6.13 (July 06, 2022) {#arcanedoc_news_changelog_version360}

### New Features/Improvements:

- Addition of an Arcane::IRandomNumberGenerator interface for a random number
  generation service (\pr{266})
- Adds support for material variables in `axl` files for the C# generator
  (\pr{273})
- Removes node connectivity allocation in old connectivities. This reduces the
  memory footprint (\pr{231}).
- Adds to the classes Arccore::Span, Arccore::ArrayView,
  Arccore::ConstArrayView, as well as views on variables, the ' operator()'
  which behaves like the 'operator[]'. This allows uniform writing
  across different containers and associated views (\pr{223}, \pr{222},
  \pr{205}).
- Adds information about the origin and dimension of the Cartesian mesh to
  Arcane::ICartesianMeshGenerationInfo (\pr{221}).
- Adds collective statistics at runtime on the time spent in message exchange
  operations. These statistics include the minimum, maximum, and average time
  for all ranges passed in these calls (\pr{220})
- Adds two additional implementations for material synchronization. Version 7
  allows for a single allocation during this synchronization, and version 8
  allows this allocation to be maintained from one synchronization to the next
  (\pr{219}).
- Adds implementation of multi-phase synchronizations, allowing the use of
  fixed-size arrays and/or processing a subset of neighbors (\pr{214}).
- Adds access for accelerators to certain Arcane::MDSpan methods (\pr{217}).
- Adds access to edge connectivities in Arcane::UnstructuredMeshConnectivityView
  (\pr{216})
- Adds an interface accessible via 'Arcane::IMesh::indexedConnectivityMng()'
  that allows for easily adding new connectivities (\pr{201}).
- Adds a new algorithm for calculating the uniqueId() of edges (Edge) for
  Cartesian meshes
- Adds support for Arccore classes for the operator `operator[]` with multiple
  arguments (\pr{241}).
- Makes calls to Arcane::Accelerator::makeQueue() thread-safe by calling the
  method Arcane::Accelerator::Runner::setConcurrentQueueCreation() (\pr{242})

### Changes:

- The classes managing materials are split into two components. One part is now
  in the `arcane_core` component. This change is normally transparent to %Arcane
  users and requires no modification of the sources (\pr{264},\pr{270},\pr{274})
- References are compacted after calling Arcane::IItemFamily::compactItems().
  This prevents the array containing internal entity information from
  unnecessarily growing. Since this change can introduce a difference in the
  order of certain operations, it is possible to disable it by setting the
  environment variable `ARCANE_USE_LEGACY_COMPACT_ITEMS` to the value `1`
  (\pr{225}).
- The types managing the `localId()` associated with entities
  (Arcane::NodeLocalId, Arcane::CellLocalId, ...) are now `typedef`s of a
  template class Arcane::ItemLocalIdT.
- The different overloads in the variable access operators (operator[]) are
  removed. You must now use an 'Arcane::ItemLocalIdT' as the indexer. Conversion
  operators to this type have been added to keep the source code compatible.
- Variables that were still allocated when calling
  Arcane::IVariableMng::removeAllVariables() are automatically unregistered.
  This prevents crashes when references to variables still existed after this
  call. This can notably happen with C# extensions because the associated
  services and modules are managed by a 'Garbage Collector' (\pr{200}).
- The use of Arcane::Timer::TimerVirtual is deprecated. Timers that use this
  property behave as if they had the Arcane::Timer::TimerReal attribute.

### Corrections:

- Fixes incorrect values in Arcane::IItemFamily::localConnectivityInfos() and
  Arcane::IItemFamily::globalConnectivityInfos() for connectivities other than
  those to nodes. This bug was introduced when transitioning to new
  connectivities (\pr{230}, \pr{27}).
- Fixes various bugs in version 3 of BasicReaderWriter (\pr{238})

### Internal:

- Uses variables to retain the Arcane::ItemInternal::owner() and
  Arcane::ItemInternal::flags() fields instead of keeping the information in
  Arcane::ItemSharedInfo. This will eventually allow the corresponding field in
  Arcane::ItemSharedInfo to be removed (\pr{227}).

Axlstar version 2.0.3.0 update:

- Adds support in 'axl' files for Arcane::IVariable::PNoExchange,
  Arcane::IVariable::PNoReplicaSync, and Arcane::IVariable::PPersistant.

%Arccore version 2.0.11.0 update:

- Adds function `mpDelete()` to destroy `IMessagePassingMng` instances
  (\pr{258})
- Optimizations in class `String`(\pr{256},\pr{247})
  - Add move constructor String(String&&) and move copy operator
    operator=(String&&)
  - Make `String` destructor inline
  - Make method `String::utf16()` deprecated (replaced by
    `StringUtils::asUtf16BE()`)
  - Methods `String::bytes()` and `String::format` no longer throws exceptions
  - Add a namespace `StringUtils` to contain utilitarian functions.
- Add support for multisubscript `operator[]` from C++23 (\pr{241})
- Add `operator()` to access values of `ArrayView`, `ArrayView2`, `ArrayView3`,
  `ArrayView4`, `Span`, `Span2` and
  `const` versions of these views (\pr{223}).
- Add `SmallSpan2` implementation for 2D arrays whose `size_type` is an
  `Int32` (\pr{223}).
- Add `SpanImpl::findFirst()` method (\pr{211})
- Fix build on Ubuntu 22.04

___

## Arcane Version 3.5.7 (April 7, 2022) {#arcanedoc_news_changelog_version350}

### New Features/Improvements:

- Adds class Arcane::SimdReal3x3 and Arcane::SimdReal2x2, which are the vector
  equivalents of Arcane::Real3x3 and Arcane::Real2x2
- Support for version 4.2 of VTK format mesh files
- Adds a new synchronization implementation that uses the `MPI_Sendrecv` call.
- Adds the possibility of using collective messages (MPI_AllToAllv) instead of
  point-to-point messages when exchanging entities following a load balance.
  This mechanism is temporarily accessible by specifying the environment
  variable
  `ARCANE_MESH_EXCHANGE_USE_COLLECTIVE` (\pr{138},\pr{154}).
- In bit-by-bit comparison, adds the possibility of only performing the
  comparison at the end of execution instead of doing it at every time step.
  This is done by specifying the environment variable STDENV_VERIF_ONLY_AT_EXIT.
- Adds a 3D honeycomb mesh generator (\pr{149}).
- Adds support for specifying the element layout in the Arcane::NumArray class.
  Two layouts are currently implemented:
  LeftLayout and RightLayout (\pr{151})
- Adds method Arcane::Accelerator::RunQueue::copyMemory() to perform
  asynchronous memory copies (\pr{152}).
- Improves ROCM/HIP support. AMD GPU support is now functionally equivalent to
  NVIDIA GPU support via Cuda (\pr{158}, \pr{159}).
- Adds support for pinned memory (Host Pinned Memory) for CUDA and ROCM
  (\pr{147}).
- Adds class 'Arcane::Accelerator::RunQueueEvent' to support events on
  'Arcane::Accelerator::RunQueue' and thus allow synchronization between
  different queues (\pr{161}).

### Changes:

- Removes the more used macros ARCANE_PROXY and ARCANE_TRACE (\pr{145})

### Corrections:

- Fixes incorrect detection of OneTBB version 2021.5 following the removal of
  the 'tbb_thread.h' file (\pr{146})
- Fixes certain missing Cartesian information when there is only one mesh layer
  in Y or Z (\pr{162}).
- Fixes missing implementation of 'Arccore::Span<T>::operator==' when the type
  `T` is not constant (\pr{163}).
- Removes some overly numerous listing messages.

### Internal:

- Uses a specific implementation for the Arcane::NumArray container instead of
  Arccore::UniqueArray2 (\pr{150}).
- Uses `Int32` instead of `Int64` to index elements in Arcane::NumArray
  (\pr{153})
- Adds `constexpr` and `noexcept` to certain Arccore classes (\pr{156}).
- Arccore version 2.0.9.0 update

___

## Arcane Version 3.4.5 (February 10, 2022) {#arcanedoc_news_changelog_version340}

### New Features/Improvements:

- In the accelerator API, Arcane::NumArray now supports directly allocating
  memory on the accelerator. Previously, only unified memory was available. The
  enumeration Arcane::eMemoryRessource and the type Arcane::IMemoryRessourceMng
  allow this to be managed (\pr{111}, \pr{113}).
- Minor improvements to documentation (\pr{117}):
  - Added relative paths for header files.
  - Added classes and types from %Arccore
- Adds a new method for calculating the uniqueId() of faces in the Cartesian
  case. This new method allows for a Cartesian numbering of faces that is
  consistent with those of nodes and meshes. To use it, you must specify the
  option `<face-numbering-version>4</face-numbering-version>` in the data set
  within the mesh generator tag (\pr{104}).
- Adds an option in the %Arcane post-processor to suppress dump output at the
  end of the calculation.
- Adds implementation of Arcane::IParallelMng::gather() and
  Arcane::IParallelMng::gatherVariable() for shared memory mode and hybrid mode
- Adds the list of media in which the material is present to
  Arcane::Materials::MeshMaterialInfo
- Supports compilation with the NVIDIA HPC SDK.
- Supports (partially) deallocating the mesh
  (Arcane::IPrimaryMesh::deallocate()), which allows it to be reallocated later.
- Adds a 2D honeycomb mesh generator.

### Changes:

- Adds the '%Arcane::' namespace to the `CMake` targets provided by %Arcane. For
  example, the 'arcane_core' target becomes 'Arcane::arcane_core'. Old names
  remain valid (\pr{120}).
- Deprecates the conversion of Arcane::ItemEnumerator to
  Arcane::ItemEnumeratorT. This prevents accidentally indexing a mesh variable
  with an enumerator of the wrong type (for example, indexing a variable in
  meshes with a node enumerator).

### Corrections:

- Fixes the 'operator=' for the 'Arcane::CellDirectionMng' class (\pr{109})
- Fixes unnecessary `Int64` to `Int32` conversion in the construction of
  Cartesian meshes, which prevented exceeding 2^31 meshes (\pr{98})
- Fixes incorrect calculation of time spent in synchronizations. Only the time
  of the last wait was used instead of the cumulative time (commit cf2cade961)
- Fixes name collision for the 'MessagePassingService' option on the command
  line (commit 15670db4)

### Internal:

- Cleanup of the Accelerator API

%Arccore version 2.0.8.1 update:

- Improve doxygen documentation for types and classes in `message_passing`
  component.
- Add functions in `message_passing` component to handle non blocking
  collectives (\pr{116}, \pr{118})
- Add some '#defines' to compile
  with [hipSYCL](https://github.com/illuhad/hipSYCL).
- Update '_clang-format' file for version 13 of LLVM/Clang.

___

## Arcane Version 3.3.0 (December 16, 2021) {#arcanedoc_news_changelog_version330}

### New Features/Improvements:

- Adds the possibility of specifying the maximum number of messages in flight
  during a load balance. For now, this is done by specifying the environment
  variable ARCANE_MESH_EXCHANGE_MAX_PENDING_MESSAGE.
- Adds the possibility of using Arcane::Real2x2 and Arcane::Real3x3 on
  accelerators
- Adds method Arcane::mesh_utils::printMeshGroupsMemoryUsage() to display the
  memory consumption associated with groups and
  Arcane::mesh_utils::shrinkMeshGroups() to resize the memory used by the groups
  to just enough
- Support for pinning threads (see \ref arcanedoc_execution_launcher)

### Changes:

- Adds the namespace Arcane::ParallelMngUtils to contain utility functions of
  Arcane::IParallelMng instead of using the virtual methods of this interface.
  The new methods replace
  Arcane::IParallelMng::createGetVariablesValuesOperation(),
  Arcane::IParallelMng::createTransferValuesOperation(),
  Arcane::IParallelMng::createExchanger(),
  Arcane::IParallelMng::createSynchronizer(),
  Arcane::IParallelMng::createTopology().
- Deprecates access to `Arccore::ArrayView<Array<T>>` in
  Arcane::CaseOptionMultiSimpleT. The method
  Arcane::CaseOptionMultiSimpleT::view() must be used instead.

### Corrections:

- Adds version 4 for ghost layer calculation, which allows calling
  Arcane::IMeshModifier::updateGhostLayers() even if one or more ghost mesh
  layers already exist.

### Internal:

- Cleanup of synchronization message management
- Begins accelerator support for ROCM/HIP (AMD) version
- Support for glibc version 2.34, which no longer contains memory management
  'hooks' (this mechanism has been obsolete for years).
- Adds the possibility of compiling with the C++20 standard.

%Arccore version 2.0.6.0 update:

- Update Array views (\pr{76})
  - Add `constexpr` and `noexcept` to several methods of `Arccore::ArrayView`,
    `Arccore::ConstArrayView` and `Arccore::Span`
  - Add converters from `std::array`
- Separate metadata from data in 'Arccore::AbstractArray' (\pr{72})
- Deprecate `Arccore::Array::clone()`, `Arccore::Array2::clone()` and make
  `Arccore::Array2` constructors protected (\pr{71})
- Add support for compilation with AMD ROCM HIP (e5d008b1b79b59)
- Add method `Arccore::ITraceMng::fatalMessage()` to throw an
  `Arccore::FatalErrorException` in a method marked `[[noreturn]]`
- Add support to compile with C++20 with `ARCCORE_CXX_STANDARD` optional CMake
  variable (665292fce)
- [INTERNAL] Add support to change return type of `IMpiProfiling` methods. These
  methods should return `int` instead of `void`
- [INTERNAL] Add methods in `MpiAdapter` to send and receive messages without
  gathering statistics
- [INTERNAL] Add methods in `MpiAdapter` to disable checking of requests. These
  checks are disabled by default if CMake variable `ARCCORE_BUILD_MODE` is
  `Release`

___

## Arcane Version 3.2.0 (November 15, 2021) {#arcanedoc_news_changelog_version320}

### New Features/Improvements:

- Adds an Arcane::IMeshPartitionerBase interface to perform only partitioning
  without support for repartitioning. The Arcane::IMeshPartitioner interface now
  inherits from Arcane::IMeshPartitionerBase.
- Adds to Arcane::MeshReaderMng the possibility of creating meshes with any
  Arcane::IParallelMng via the method Arcane::MeshReaderMng::readMesh().
- Adds an Arcane::IGridMeshPartitioner interface to partition a mesh according
  to a grid. A service named `SimpleGridMeshPartitioner` implements this
  interface. The page \ref arcanedoc_entities_snippet_cartesianmesh shows an
  example of use.

### Changes:

- CMake version 3.18 on Unix machines and CMake 3.21 on Windows.
- Deprecates the `singleton()` method in Arcane::ItemTypeMng. Instances of this
  class are attached to the mesh and can be retrieved via
  Arcane::IMesh::itemTypeMng().
- Moves the classes managing the Cartesian mesh to the `arcane/cartesianmesh`
  directory. Old paths in `arcane/cea` remain valid.
- By default, uses version 3 (instead of 2) of the ghost mesh creation service.
  This version is more efficient when using a large number of subdomains because
  it uses collective communications.
- Removes memory preallocation for old connectivities.
- Makes the constructors of Arcane::ItemSharedInfo private to %Arcane
- Raises a fatal exception if task support is requested but no implementation is
  available. Previously, there was only a warning message.

### Corrections:

- Fixes crash (SEGV) when using tasks and subtasks sequentially.

___

## Arcane Version 3.1.2 (October 21, 2021) {#arcanedoc_news_changelog_version310}

### New Features/Improvements:

- New implementation of mesh graphs using `Arcane::DoF`.
- Adds the possibility of renumbering (via the method
  Arcane::ICartesianMesh::renumberItemsUniqueId()) entities in AMR meshes by
  patch to have the same numbering regardless of the decomposition.
- Documentation update for accelerators

### Changes:

- The mesh reader for the `GMSH` format is now in the `arcane_std` library
  instead of `arcane_ios`. Therefore, there is no longer a need to perform
  linking with the latter to be able to read meshes of this format.
- Removal of old entity types `Link` and `DualNode` and associated enumerations
  and classes
- Removal of certain classes associated with old connectivities
- Removes support for the RedHat 6 operating system.

### Corrections:

- Fixes crash when updating materials if the global variable associated with a
  material is deallocated (Arcane::IVariable::isUsed()==false)
- Fixes floating point exception (FPE) with versions 2.9.9+ of `libxml2`. This
  library explicitly performs division by 0 during initialization.

___

## Arcane Version 3.0.5 (September 30, 2021) {#arcanedoc_news_changelog_version305}

### New Features/Improvements:

- Moves the Arcane::NumArray class to the `utils` component. This makes it
  accessible outside of its usage for accelerators
- Various modifications in Arcane::NumArray and associated classes (notably
  Arcane::MDSpan) to make them more generic and for future use in %Arcane
  variables.
- Simplifies and extends the use of Arcane::UnstructuredMeshConnectivityView
- Adds method 'IVariable::dataFactoryMng()' to retrieve the
  Arcane::IDataFactoryMng associated with the variable data.
- Adds methods Arcane::Real2::normL2(), Arcane::Real3::normL2(),
  Arcane::Real2::squareNormL2() and Arcane::Real3::squareNormL3() to replace the
  'abs()' and 'abs2()' methods of these two classes.
- Adds methods Arcane::Real2::absolute(), Arcane::Real3::absolute(), to return a
  vector with the absolute values per component.
- Adds support for OneTBB version 2021.
- Adds macros RUNCOMMAND_ENUMERATE() and RUNCOMMAND_LOOP() to iterate over
  accelerators
- Adds class Arcane::Accelerator::IAcceleratorMng to retrieve information for
  using accelerators. This interface allows retrieving the default execution
  environment and the default execution queue.
- Adds class Arcane::StandaloneAcceleratorMng to use accelerators without
  initializing an application.
- Adds support for multi-threading nested loops up to 4 levels (\pr{10})

### Changes:

- Deprecates the internal class Arcane::IDataFactory and corresponding methods
- Deprecates the methods Arcane::IDataFactoryMng::createEmptySerializedDataRef()
  and Arcane::IDataFactoryMng::createSerializedDataRef()
- Removes the obsolete methods Arcane::IData::clone() and
  Arcane::IData::cloneTrue().

### Corrections:

- Fixes the copy operator when using two views of the same type. Since the copy
  operator was not overloaded, only the reference was modified and not the
  value. This occurred in the following case:
  using namespace Arcane; auto v1 = viewInOut(var1); auto v2 = viewInOut(var2);
  ENUMERATE_CELL(icell,allCells()){ v2[icell] = v1[icell]; // ERROR: v2 then
  referred to v1. }
- Fixes compilation error in the constructor Span<const T> from a
  'ConstArrayView'.
- [arccore] Fixes missing message sending when calling
  Arccore::MessagePassing::PointToPointSerializerMng::waitMessages(). The call
  to
  Arccore::MessagePassing::PointToPointSerializerMng::processPendingMessages()
  was missing. Because of this bug, the class
  Arcane::TransferValuesParallelOperation did not work and consequently the
  method Arcane::IItemFamily::reduceFromGhostItems() did not work either.
- [config] Supports the case where multiple versions of the 'dotnet' SDK are
  installed. In this case, the most recent version is used.

___

## Arcane Version 3.0.3 (Not released) {#arcanedoc_news_changelog_version303}

### New Features/Improvements:

- Support for parallel AMR by patch
- Adds an Arcane::SimpleSVGMeshExporter class to export a set of meshes in SVG
  format
- Support for patch AMR in the Arcane::DirNode class for neighboring meshes by
  direction.
- During group synchronization, ensures that all subdomains have the same groups
  and that synchronization occurs in the same order.

### Changes:

- Deprecates the methods Arcane::IArrayDataT::value() and
  Arcane::IArray2DataT::value(). Instead, you can use the methods
  Arcane::IArrayDataT::view() and Arcane::IArray2DataT::view(). The purpose of
  these changes is to be able to hide the container used for the implementation
- Adds methods Arcane::arcaneParallelFor() and Arcane::arcaneParallelForeach()
  to replace the various methods Arcane::Parallel::For() and
  Arcane::Parallel::Foreach().

### Corrections:

- In patch AMR, ensures that neighboring entities by direction are always in the
  same patch level.
- Fixes some missing dependencies during compilation that could cause
  compilation errors in certain cases.
- Fixes compilation errors in examples outside the source directory.

___

## Arcane Version 3.0.1 (May 27, 2021) {#arcanedoc_news_changelog_version301}

This version is the first 'open source' version of %Arcane.

### New Features/Improvements:

- New version of the Arcane::BasicReaderWriter read/write service to generate
  fewer files and support compression. This service can be used for both
  backups/restorations and bit-by-bit comparison of variables. The C# variable
  comparison utility has been updated to support this new version.
- Support for 'msh' format mesh files version 4.1. This version allows
  specifying groups of faces or meshes in the mesh file.
- Internally, uses a single executable for all C# utilities.

### Changes:

- Adds the possibility during %Arcane compilation to specify required packages
  and not search for default packages.
