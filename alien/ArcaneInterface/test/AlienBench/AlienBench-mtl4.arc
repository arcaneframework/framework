<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="AlienBench" xml:lang="en">
  <arcane>
    <title>Test unitaires des solveurs lineaires</title>
    <timeloop>BenchLoop</timeloop>
  </arcane>

  <arcane-post-processing>
    <output-period>1</output-period>
    <output>
      <variable>U</variable>
      <variable>X</variable>
      <variable>K</variable>
      <variable>S</variable>
    </output>
  </arcane-post-processing>

  <mesh>
    <meshgenerator>
      <cartesian>
        <origine>0. 0. 0.</origine>
        <nsd>1 1 1</nsd>
        <lx nx="10">1.</lx>
        <ly ny="10">1.</ly>
        <lz nz="10">1.</lz>
      </cartesian>
    </meshgenerator>
  </mesh>


    <alien-bench>
      <!-- big diagonal-coefficient keep diagonal dominant matrix -->
      <diagonal-coefficient>1.</diagonal-coefficient>
      <!--lambdax>0.125</lambdax>
      <lambday>0.25</lambday>
      <alpha>10.</alpha>
      <sigma>1000000.</sigma>
      <epsilon>0.01</epsilon-->
      <homogeneous>true</homogeneous>
      <zero-rhs>false</zero-rhs>
      <nb-resolutions>1</nb-resolutions>

      <linear-solver name="MTLSolver">
        <solver>CG</solver>
        <preconditioner>Diag</preconditioner>
        <verbose>true</verbose>
        <output-level>2</output-level>
      </linear-solver>

  </alien-bench>
</case>
