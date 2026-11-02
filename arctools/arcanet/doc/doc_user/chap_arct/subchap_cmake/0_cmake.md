# CMake functions {#arcanetdoc_arct_cmake}

In order to generate ARCT files and convert ARCT files into CTest tests, we have written CMake functions
to be used in projects using CMake.

It is recommended to have read the chapter on the ARCT format before.

First, CMake functions are provided to build the ARCT file piece by piece.

Then, a CMake function makes it possible to convert an ARCT file into CTest tests.

<br>

Contents of this subchapter:

1. \subpage arcanetdoc_arct_cmake_arct_writer <br>
   ARCT file writer in CMake.

2. \subpage arcanetdoc_arct_cmake_arct_to_ctest <br>
   ARCT file to CTest converter.

Some examples to use the CMake writer:

1. \subpage arcanetdoc_arct_cmake_example1 <br>
   A minimal file: only a `general` block.

2. \subpage arcanetdoc_arct_cmake_example2 <br>
   Base cases.

3. \subpage arcanetdoc_arct_cmake_example3 <br>
   Common configurations and dependencies.

4. \subpage arcanetdoc_arct_cmake_example4 <br>
   Resolved variations.

5. \subpage arcanetdoc_arct_cmake_example5 <br>
   Unresolved variations.

6. \subpage arcanetdoc_arct_cmake_example6 <br>
   Variant limiters (`!` and `+`).

7. \subpage arcanetdoc_arct_cmake_example7 <br>
   Variations of variations.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_variations4
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cmake_arct_writer
</span>
</div>
