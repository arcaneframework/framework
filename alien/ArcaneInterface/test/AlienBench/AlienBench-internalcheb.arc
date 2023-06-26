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
      <redistribution>false</redistribution>
      <!-- big diagonal-coefficient keep diagonal dominant matrix -->
      <homogeneous>true</homogeneous>
      <diagonal-coefficient>0.</diagonal-coefficient>
      <lambdax>0.125</lambdax>
      <lambday>0.25</lambday>
      <alpha>10.</alpha>
      <sigma>1000000.</sigma>
      <epsilon>0.01</epsilon>

      <alien-core-solver>
        <backend>SimpleCSR</backend>
        <solver>BCGS</solver>
        <preconditioner>ChebyshevPoly</preconditioner>
        <poly-order>1</poly-order>
        <poly-factor>30.</poly-factor>
        <poly-factor-max-iter>10</poly-factor-max-iter>
        <max-iter>1000</max-iter>
        <tol>1e-8</tol>
        <output-level>3</output-level>
        <asynch>0</asynch>
      </alien-core-solver>
      <!--linear-solver name="TrilinosSolver">
        <solver>BiCGStab</solver>
        <max-iteration-num>5</max-iteration-num>
        <stop-criteria-value>1e-8</stop-criteria-value>
        <preconditioner>Chebyshev</preconditioner>
        <nb-threads>1</nb-threads>
        <output>1</output>
      </linear-solver-->
      <linear-solver name="HTSSolver">
        <solver>BiCGStab</solver>
        <max-iteration-num>1000</max-iteration-num>
        <stop-criteria-value>1e-8</stop-criteria-value>
        <preconditioner>Chebyshev</preconditioner>
        <poly-degree>1</poly-degree>
        <poly-factor>30.</poly-factor>
        <poly-factor-max-iter>10</poly-factor-max-iter>
        <output>3</output>
        <nb-part>1</nb-part>
        <nb-subpart>1</nb-subpart>
      </linear-solver>


    
  </alien-bench>
</case>
