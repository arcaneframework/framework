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
        <nsd>2 1 1</nsd>
        <lx nx="100">1.</lx>
        <ly ny="10">1.</ly>
        <lz nz="10">1.</lz>
      </cartesian>
    </meshgenerator>
  </mesh>


    <alien-bench>
      <use-accelerator>false</use-accelerator>
      <redistribution>false</redistribution>
      <block-size>12</block-size>
      <homogeneous>true</homogeneous>
      <!-- big diagonal-coefficient keep diagonal dominant matrix -->
      <diagonal-coefficient>10.0</diagonal-coefficient>
      <lambdax>0.125</lambdax>
      <lambday>0.25</lambday>
      <alpha>10.</alpha>
      <sigma>1000000.</sigma>
      <epsilon>0.01</epsilon>
      <zero-rhs>false</zero-rhs>
      <nb-resolutions>1</nb-resolutions>
      <linear-solver name="AlienCoreSolver">
        <backend>SimpleCSR</backend>
        <solver>BCGS</solver>
        <preconditioner>FILU0</preconditioner>
        <max-iter>1000</max-iter>
        <tol>1.e-3</tol>
        <output-level>3</output-level>
      </linear-solver>
  </alien-bench>
</case>
