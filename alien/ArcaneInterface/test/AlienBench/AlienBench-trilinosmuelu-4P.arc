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
        <nsd>4 1 1</nsd>
        <lx nx="10">1.</lx>
        <ly ny="10">1.</ly>
        <lz nz="10">1.</lz>
      </cartesian>
    </meshgenerator>
  </mesh>


    <alien-bench>
      <redistribution>false</redistribution>
      <!-- big diagonal-coefficient keep diagonal dominant matrix -->
      <homogeneous>true</homogeneous>
      <diagonal-coefficient>0.</diagonal-coefficient>
      <lambdax>0.125</lambdax>
      <lambday>0.25</lambday>
      <alpha>10.</alpha>
      <sigma>1000000.</sigma>
      <epsilon>0.01</epsilon>
      <unit-rhs>1</unit-rhs>

      <linear-solver name="TrilinosSolver">
        <solver>BiCGStab</solver>
        <max-iteration-num>1000</max-iteration-num>
        <stop-criteria-value>1e-8</stop-criteria-value>
        <preconditioner>MueLu</preconditioner>
        <!--chebyshev>
           <degree>3</degree>
        </chebyshev!-->
        <relaxation>
           <type>Symmetric Gauss-Seidel</type>
           <sweeps>2</sweeps>
           <damping-factor>0.9</damping-factor>
        </relaxation>
        <muelu>
           <verbosity>high</verbosity>
           <symmetric>false</symmetric>
           <max-level>20</max-level>
           <cycle-type>W</cycle-type>
           <smoother-type>RELAXATION</smoother-type>
           <coarse-type>KLU2</coarse-type>
           <multigrid-algorithm>pg</multigrid-algorithm>
        </muelu>
        <iluk>
          <level-of-fill>0</level-of-fill>
        </iluk>
        <output>1</output>
      </linear-solver>

    
  </alien-bench>
</case>
