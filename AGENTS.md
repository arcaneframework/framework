# Arcane Framework — Agent Instructions

## Repository Overview

Arcane is a development platform for **massively parallel unstructured 2D/3D mesh computation codes**. Written by CEA/IFPEN, it provides mesh management, data structures, parallelism (MPI/threads), I/O, and a service/plugin architecture for scientific computing applications. The repository is a CMake-based monorepo.

## Repo Structure

| Directory | Purpose |
|-----------|---------|
| `arccore/` | Core library — concurrency, serialization, mesh geometry, accelerators, message passing |
| `arcane/` | Main framework — mesh management, I/O, services, driver, test infrastructure |
| `alien/` | Linear algebra interface (Hypre, PETSc, Trilinos) with Arcane coupling |
| `arccon/` | Custom CMake build-system layer (macros, install dirs, .NET helpers) — no C++ code |
| `axlstar/` | AXL XML-to-C++/C# code generator (.NET/`dotnet` app) |
| `arctools/` | Documentation generator (`adoc/`) and Neo CLI tools |
| `dependencies/` | Git submodule — NuGet packages + ArcDependencies CMake finder scripts |

**Build order** (automatic via top-level CMake): `arccon` → `axlstar` → `arccore` → `arcane` → `alien`.

---

## Component Details

### arccore — Core Library

Foundation for all Arcane components. Organized into CMake components:

| Component | Purpose |
|-----------|---------|
| `base` | Core types, exceptions, error handling, memory management |
| `common` | Math utilities, string handling, time, hash, UUID |
| `collections` | Containers (DynamicArray, HashTable, etc.) |
| `concurrency` | Thread pools, barriers, atomics |
| `serialize` | Serialization framework for all Arcane types |
| `trace` | Logging and tracing system |
| `message_passing` / `message_passing_mpi` | MPI and shared-memory communication |
| `accelerator` / `accelerator_native` | Unified GPU accelerator API (CUDA/HIP/SYCL) |

- **C++20+** required. Minimum: GCC 11 or Clang 16.
- Components are added via `arccore_add_component_directory(<name>)`.
- Generated config headers: `build/arccore/arccore_config.h`, `build/arccore_version.h`.

### arcane — Main Framework

The top-level application framework built on arccore. Key sub-libraries:

| Library | Purpose |
|---------|---------|
| `arcane_core` | Core runtime, service/module base classes, AXL processing |
| `arcane_mesh` | Mesh entities (nodes, edges, faces, cells), connectivities, ghost layers |
| `arcane_impl` | Implementation utilities |
| `arcane_utils` | Common utilities |
| `arcane_ios` | I/O readers/writers (VTK, MED, Gmsh, XMF) |
| `arcane_driver` | `.arc` case-file driver |
| `arcane_geometry` | Geometric computations (ray intersection, etc.) |
| `arcane_hdf5` | HDF5 I/O support |
| `arcane_parallel` | MPI and thread parallelism wrappers |

Additional modules:
- `std/` — Standard services: checkpoint, mesh partitioning (Metis/PTScotch/Zoltan), mesh generators (Sod, Cartesian, HoneyComb), I/O services, post-processing, profilers (PAPI, OTF2)
- `accelerator/` — GPU-accelerated mesh operations. Provides mesh-aware GPU abstractions built on top of arccore's low-level accelerator primitives:
  - `RunCommandLoop` / `RunCommandEnumerate` — kernel launch commands that iterate over mesh entities (cells, faces, nodes) with device-side views
  - `VariableViews` / `MaterialVariableViews` — zero-copy device pointers into Arcane mesh variables, enabling kernels to read/write mesh data directly on GPU
  - `Views.h` / `SpanViews.h` — device-compatible array views for work-item-local data
  - `NumArray.h` — GPU-resident numerical arrays with host/device synchronization
  - Generic algorithms: `Filter` (selective copy), `Reduce` (parallel reduction), `Scan` (exclusive/inclusive scan), `Sort` (key-value sort), `Partitioner` (work partitioning)
  - `Atomic.h`, `LocalMemory.h`, `RunQueue.h` — low-level GPU synchronization and memory management
  - Backend-specific directories: `cuda/`, `hip/`, `sycl/` for runtime-specific code
- `aleph/` — Numerical linear algebra components
- `corefinement/` — Mesh refinement/coarsening
- `launcher/` — Parallel launch infrastructure
- `lima/` — Lima mesh format support
- `driver/` — `.arc` case-file execution driver
- `materials/` — Multi-constituent material support
- `cartesianmesh/` — Cartesian mesh with AMR

**Public API boundary:** Only `core/`, `materials/`, `utils/`, `launcher/`, `accelerator/`, `cartesianmesh/`, and `hdf5/` are public API. All other `arcane/` subdirectories (`mesh/`, `ios/`, `driver/`, `geometry/`, `std/`, `aleph/`, `corefinement/`, `lima/`, `parallel/`, etc.) are private — they may be installed to `include/` but are not stable and should not be used by external code.

### alien — Linear Algebra Interface

Provides a unified interface for large sparse linear solvers:

- **`standalone/`** — Standalone Alien library (C++17, MPI, BLAS). Plugins for Hypre, Trilinos, PETSc, Ginkgo.
- **`ArcaneInterface/`** — Adapter layer connecting Alien to Arcane meshes and variables.

### arccon — Build System Layer

Pure CMake project (no C++). Provides:
- `find_package(Arccon)` — dependency finder
- `arccon_install_directory()` / `arccon_dotnet_install_publish_directory()` — install helpers
- `.NET/msbuild` integration macros (`ArcconDotNet.cmake`)
- Standard install directory layout (`ArcconSetInstallDirs.cmake`)

Minimum version: CMake 3.18. Loaded via `find_package(Arccon REQUIRED)`.

### axlstar — AXL Code Generator

.NET application (`Arcane.Axl.sln`) that processes `.axl` XML descriptor files. Produces:
- `axl2cc` / `axl2ccT4` — generates C++ header files with factory/serialization code from AXL
- `axldoc` — documentation generator
- `axlcopy` — AXL file copier

Requires `dotnet` (coreclr). The generated code includes C++ classes, constructors, and serialization logic for Services and Modules defined in AXL files.

---

## AXL Code Generation

**AXL** (Arcane XML Language) files (`*.axl`) in `src/` and `tests/` describe Services and Modules. At build time:

1. `axlstar` compiles the .NET solution to produce `axl2ccT4`
2. `axl2ccT4` processes each `.axl` file and generates `<name>_axl.h` headers
3. Generated headers are included by C++ source files — they contain factory registration, property accessors, and serialization glue

**Key files:**
- `Arcane.Axl/axl.xsd` — AXL schema (installed to `share/axl.xsd`)
- `build/bin/` — generated executables (`axl2cc`, `axl2ccT4`, etc.)
- `build/lib/` — generated `_axl.h` headers

**Adding a new Service/Module:** create a `.axl` descriptor and the corresponding `.cc` implementation. The build system auto-generates the glue code.

---

## Build Rules

- **Out-of-source only.** In-source build aborts with `FATAL_ERROR`.
- **Shared libs mandatory.** `BUILD_SHARED_LIBS=TRUE` enforced; static does not work.
- **Clone submodules first:**
  ```
  git clone --recurse-submodules https://github.com/arcaneframework/framework
  ```
- **Standard configure/build/install:**
  ```
  cmake -S /path/to/sources -B /path/to/build
  cmake --build /path/to/build
  cmake --install /path/to/build
  ```
- **Select components:** `cmake -DARCANEFRAMEWORK_BUILD_COMPONENTS=Arcane -S . -B build`
  Valid values: `Arcane`, `Alien`, `Arcane;Alien` (default).
- **Deprecated legacy flag:** `FRAMEWORK_BUILD_COMPONENT=all|arcane|arccore|arccon|axlstar|alien|alien_standalone|neo|doc`

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `ARCCORE_CXX_STANDARD` | `20` | C++ standard: `20`, `23`, or `26` |
| `ARCCORE_ACCELERATOR_MODE` | — | GPU backend: `CUDA`, `HIP`, or `SYCL` |
| `ARCANEFRAMEWORK_BUILD_COMPONENTS` | `Arcane;Alien` | Which framework components to build |
| `ARCANE_ENABLE_TESTS` | `ON` | Enable test compilation |
| `ARCANE_ENABLE_DOTNET_WRAPPER` | `ON` | C#/.NET wrapper (SWIG) |
| `ARCANE_ENABLE_ALEPH` | `ON` | Numerical algebra component |
| `ARCANE_DEFAULT_PARTITIONER` | `Auto` | Default mesh partitioner |
| `ARCCORE_BUILD_MODE` | — | `Debug`, `Check`, or `Release` |
| `ARCANE_DISABLE_PERFCOUNTER_TESTS` | `OFF` | Disable perf/event tests (CI) |

**Build modes:** `Debug` (full assertions), `Check` (assertions without debug overhead), `Release` (no assertions). `Check` maps to `CMAKE_BUILD_TYPE=Release` with internal check macros enabled.

**CMake presets:** `CMakePresets.json` provides named presets — `cmake --preset ArcaneCuda`, `cmake --preset Arccore`, etc.

**Environment overrides:**
- `ARCANE_PACKAGE_FILE=<path>` — include an extra `.cmake` file during config
- `ARCANE_EXTRA_LIBS=lib1;lib2` — extra libraries to link

---

## Service and Module Architecture

Arcane uses a **plugin/service** architecture:

- **Service** — an interface (`*.axl` + `I*.h`) and implementation (`*.cc`) with separated interface/implementation
- **Module** — a collection of Services and data, registered at runtime
- Services are discovered and instantiated via the AXL-generated factory code
- The `.arc` case files select which modules/services to activate at runtime
- External libraries can provide services compiled outside the main tree (via `ARCANE_EXTERNAL_LIBRARIES`)

---

## Testing System

### How Tests Work

1. C++ test executables are compiled from `arcane/src/arcane/tests/` (`.cc` files + `.axl` descriptors)
2. Each test is driven by a `.arc` case file in `arcane/tests/` — an XML-like configuration that specifies:
   - Mesh generator to use (Sod, Cartesian, Simple, etc.)
   - Modules and services to activate
   - Test parameters
3. The `arcane_test_driver` executable reads the `.arc` file and executes the test
4. Some tests compare outputs using CSV comparators or binary hash comparison

### Running Tests

```bash
# Run all tests
ctest --test-dir build

# Run a single test
ctest -R testname --test-dir build

# MPI tests require root for CI runners
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```

### Special Cases

- **Performance counter tests** need `perf_event_open` syscall; disable with `-DARCANE_DISABLE_PERFCOUNTER_TESTS=ON`
- **CUDA googletest** — skip when no GPU: `-DARCANE_EXECUTE_ACCELERATOR_GOOGLETEST=OFF`
- **Samples** — separate build targets: `cmake --build build --target samples_configure && cmake --build build --target samples_build`

---

## Accelerator Support

GPU acceleration is configured at the arccore level:

| Option | Values | Notes |
|--------|--------|-------|
| `ARCCORE_ACCELERATOR_MODE` | `CUDA`, `HIP`, `SYCL` | Set before configure |
| `CMAKE_CUDA_ARCHITECTURES` | `75 80` (default) | Required for CUDA C++20 (CMake 3.26+) |
| `CMAKE_HIP_COMPILER` | `amdclang++` or `clang++` | Set for HIP builds |

**CUDA/HIP** code compiles to device kernels; **SYCL** is experimental. The unified Accelerator API in arccore abstracts the backend — arcane-level code uses `ARCANE_HAS_ACCELERATOR` and `ARCANE_ACCELERATOR_RUNTIME` to conditionally compile GPU paths.

---

## Style Conventions

- **Indentation:** 2 spaces for `.c/.cc/.h/.H/.cs`, `.cmake`, `CMakeLists.txt`
- **Encoding:** UTF-8-BOM for C/C++/C++ headers; UTF-8 for everything else
- **EditorConfig:** `.editorconfig` is in effect — trailing whitespace trimmed, final newline inserted
- **Hidden visibility:** `-fvisibility=hidden -fvisibility-inlines-hidden` on GCC/Clang

---

## Important Constraints (Easy to Miss)

- **`CMAKE_INCLUDE_CURRENT_DIR=FALSE`** in both arccore and arcane — never use `#include "foo.h"` expecting current-dir lookup; always use `#include "arccore/base/Foo.h"` style paths
- **`CMAKE_NO_SYSTEM_FROM_IMPORTED=1`** — imported package headers are NOT system headers (prevents CPATH conflicts but means `-isystem` suppression is off)
- **Tests must be added after all library subdirectories** — test `add_subdirectory()` calls happen late in `arcane/CMakeLists.txt` because they need access to generated config
- **`dependencies` submodule must exist** — version file must match `ArcDependencies_VERSION` (1.11.0+), otherwise CMake fails with a clear error
- **Linker `--no-as-needed`** is added on Linux for `arcane_full` — without it, plugins loaded at runtime may have missing symbols
- **C# wrapper requires dotnet** — if `ARCANE_ENABLE_DOTNET_WRAPPER=ON`, a working `dotnet` SDK is needed to build `axlstar`

---

## Build Artifacts and Generated Files

| Location | Contents |
|----------|----------|
| `build/bin/` | Generated executables (`axl2cc`, test drivers, etc.) |
| `build/lib/` | Generated `_axl.h` headers, `arcane_core_config.h`, `arcane_packages.h`, `arcane_version.h` |
| `build/share/axl.xsd` | AXL schema |
| `build/share/axl/` | AXL file staging directory |

Generated headers are installed to `include/` alongside source headers.
