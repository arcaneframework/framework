# Compilation and Launch {#arcanedoc_examples_concret_example_build}

[TOC]

This last part will be quite short given that there are no major differences
from the HelloWorld CMakeLists.txt of the previous chapter
(\ref arcanedoc_examples_simple_example_cmake).
Furthermore, the compilation commands do not change significantly either
(\ref arcanedoc_examples_simple_example_build).

## CMakeLists.txt {#arcanedoc_examples_concret_example_build_cmakeliststxt}

Here is the Quicksilver CMakeLists.txt:
```cmake
cmake_minimum_required(VERSION 3.16)
project(Quicksilver LANGUAGES CXX)

set(BUILD_SHARED_LIBS TRUE)

find_package(Arcane REQUIRED)

add_executable(Quicksilver
  main.cc
  qs/QSModule.cc qs/QS_axl.h
  sampling_mc/SamplingMCModule.cc sampling_mc/SamplingMC_axl.h
  tracking_mc/TrackingMCModule.cc tracking_mc/TrackingMC_axl.h
  rng/RNGService.cc rng/RNG_axl.h
  tracking_mc/NuclearData.cc)

arcane_generate_axl(qs/QS)
arcane_generate_axl(sampling_mc/SamplingMC)
arcane_generate_axl(tracking_mc/TrackingMC)
arcane_generate_axl(rng/RNG)


arcane_add_arcane_libraries_to_target(Quicksilver)
target_compile_options(Quicksilver PUBLIC -Wpedantic)
target_include_directories(Quicksilver PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
configure_file(Quicksilver.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
```

Let's focus on the new lines.

```cmake
set(BUILD_SHARED_LIBS TRUE)
```
This line asks CMake to generate dynamic libraries instead of static libraries.

____

```cmake
target_compile_options(Quicksilver PUBLIC -Wpedantic)
```
This line allows adding a compilation option. The option added here requests the
addition of more warnings during compilation.


## Compilation {#arcanedoc_examples_concret_example_build_commands}

For compilation, we can reuse the commands presented in this chapter:
\ref arcanedoc_examples_simple_example_build, without forgetting to change
`HelloWorld` to `quicksilver`.

The list of .arc examples is available in the `Quicksilver` readme.md.

____

If a problem exists in this subsection, you can open an issue on the %Arcane
GitHub.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_concret_example_rng
</span>
<!-- <span class="next_section_button">
\ref 
</span> -->
</div>
