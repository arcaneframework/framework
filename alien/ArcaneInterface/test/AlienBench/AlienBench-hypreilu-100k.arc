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
        <lx nx="100">1.</lx>
        <ly ny="100">1.</ly>
        <lz nz="10">1.</lz>
      </cartesian>
    </meshgenerator>
  </mesh>


  <alien-bench>
    <use-accelerator>false</use-accelerator>
    <!-- big diagonal-coefficient keep diagonal dominant matrix -->
    <diagonal-coefficient>0.01</diagonal-coefficient>
    <lambdax>0.125</lambdax>
    <lambday>0.25</lambday>
    <alpha>100.</alpha>
    <sigma>1000000.</sigma>
    <epsilon>0.01</epsilon>
    <homogeneous>true</homogeneous>
    <zero-rhs>false</zero-rhs>
    <nb-resolutions>1</nb-resolutions>
    <!--alien-core-solver>
      <backend>SYCL</backend>
      <solver>BCGS</solver>
      <preconditioner>Diag</preconditioner>
      <max-iter>1000</max-iter>
      <tol>1.e-6</tol>
      <output-level>1</output-level>
    </alien-core-solver-->
    
     <linear-solver name="HypreSolver">
        <exec-space>Host</exec-space>
        <memory-type>Host</memory-type>
        <solver>BiCGStab</solver>
        <num-iterations-max>1000</num-iterations-max>
        <stop-criteria-value>1e-12</stop-criteria-value>
        <preconditioner>BJ-ILUK</preconditioner>
        <verbose>true</verbose>
      </linear-solver>
  </alien-bench>
</case>
