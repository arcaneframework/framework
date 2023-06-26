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
        <origine>0 0 0</origine>
        <nsd>2 2 1</nsd>
        <lx nx="10">1.</lx>
        <ly ny="10">1.</ly>
        <lz nz="10">1.</lz>
      </cartesian>
    </meshgenerator>
  </mesh>


    <alien-bench>
      <redistribution>false</redistribution>
      <homogeneous>false</homogeneous>
      <unit-rhs>false</unit-rhs>
      <!-- big diagonal-coefficient keep diagonal dominant matrix -->
      <diagonal-coefficient>0.</diagonal-coefficient>
      <lambdax>0.125</lambdax>
      <lambday>0.25</lambday>
      <alpha>10.</alpha>
      <sigma>1000000.</sigma>
      <epsilon>0.01</epsilon>

      <linear-solver name="HPDDMSolver">
        <max-iteration-num>1000</max-iteration-num>
        <stop-criteria-value>1e-8</stop-criteria-value>
        <geneo-nu>5</geneo-nu>
        <output-level>1</output-level>
      </linear-solver>

    
  </alien-bench>
</case>
