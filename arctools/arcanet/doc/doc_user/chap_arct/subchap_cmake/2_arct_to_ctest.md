# ARCT -> CTest converter {#arcanetdoc_arct_cmake_arct_to_ctest}

[TOC]

To be able to run the different cases of an ARCT file, a CMake function has been added to read an ARCT
file and add one test per use case.


--------------------------

### arct_to_ctest

```cmake
arct_to_ctest(ARCT_PATH <arct_path> ARCT_CONFIG_PATH <arct_config_path> CTEST_NAME_PREFIX <prefix> ARCCHER_PATH <arccher_path>)
```

This function reads an ARCT file and creates one test per use case.

The test names correspond to the complete names of each use case.

Four `oneValueArgs` arguments are available:

- `ARCT_PATH <arct_path>` (optional): gives the location of the ARCT file to read. The default value is
  `${CMAKE_BINARY_DIR}/testlist.arct`.
- `ARCT_CONFIG_PATH <arct_config_path>` (optional): gives the location of a second ARCT file to read.
  Only the `general` block will be read. It will be read first.
- `CTEST_NAME_PREFIX <prefix>` (optional): gives a prefix to add to the CTest test names.
- `ARCCHER_PATH <arccher_path>` (optional): gives the path of the ArcCher executable. It will be set by
  default by Arcane, if one uses the ArcaNet of the Arcane installation.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cmake_arct_writer
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cmake_arct_writer_example1
</span>
</div>
