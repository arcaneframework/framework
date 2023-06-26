<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="AlienTest" xml:lang="en">
  <arcane>
    <title>Test unitaires des solveurs lineaires</title>
    <timeloop>TestLoop</timeloop>
  </arcane>

  <arcane-post-processing>
    <output-period>1</output-period>
    <output>
      <variable>U</variable>
    </output>
  </arcane-post-processing>

  <mesh>
    <!--file internal-partition='true'>cube3D.vt2</file-->
     <file internal-partition='true'>tube5x5x100.vtk</file>
     <!--file internal-partition='true'>tube2x2x4.vtk</file-->
  </mesh>


    <alien-test>
      <!-- big diagonal-coefficient keep diagonal dominant matrix -->
      <diagonal-coefficient>4</diagonal-coefficient>
      <stencil-by>node</stencil-by>
      <vect-size>1</vect-size>

      <check-memory>false</check-memory>
      <building-only>false</building-only>

      <repeat-loop>1</repeat-loop>
      <extra-equation-count>0</extra-equation-count>

      <builder>DirectBuilder</builder>

      <linear-solver name="MTLSolver">
      	<solver>CG</solver>
        <preconditioner>None</preconditioner>
      </linear-solver>
  </alien-test>
</case>
