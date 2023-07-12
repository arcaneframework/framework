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
        <origine></origine>
        <nds>1 1 1</nds>
        <lx nx="40">1.</lx>
        <ly ny="40">1.</ly>
        <lz nz="40">1.</lz>
      </cartesian>
    </meshgenerator>
  </mesh>


    <alien-bench>
      <!-- big diagonal-coefficient keep diagonal dominant matrix -->
      <diagonal-coefficient>0.</diagonal-coefficient>
      <lambdax>0.125</lambdax>
      <lambday>0.25</lambday>
      <alpha>10.</alpha>
      <sigma>1000000.</sigma>
      <epsilon>0.01</epsilon>

      <linear-solver name="PETScSolver">
        <solver name="SuperLU">
        </solver>
       <verbose>high</verbose>
      </linear-solver>

    
  </alien-bench>
</case>
