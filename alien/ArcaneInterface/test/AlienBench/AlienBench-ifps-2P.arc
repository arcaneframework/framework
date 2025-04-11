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
    </output>
  </arcane-post-processing>

  <mesh>
    <!--file internal-partition='true'>cube3D.vt2</file-->
     <!--file internal-partition='true'>tube5x5x100.vtk</file-->
     <!--file internal-partition='true'>tube2x2x4.vtk</file-->
     
    <meshgenerator>
      <cartesian>
        <origine>0. 0. 0.</origine>
        <nsd>2 1 1</nsd>
        <lx nx="10">1.</lx>
        <ly ny="10">1.</ly>
        <lz nz="10">1.</lz>
      </cartesian>
    </meshgenerator>
  </mesh>


    <alien-bench>
      <diagonal-coefficient>0.</diagonal-coefficient>
      <lambdax>0.125</lambdax>
      <lambday>0.25</lambday>
      <alpha>10.</alpha>
      <sigma>1000000.</sigma>
      <epsilon>0.01</epsilon>
      <homogeneous>true</homogeneous>
      <!--zero-rhs>true</zero-rhs-->
      <nb-resolutions>100</nb-resolutions>


      <linear-solver name="IFPSolver">
          <num-iterations-max>1000</num-iterations-max>
          <stop-criteria-value>1e-8</stop-criteria-value>
          <precond-option>ILU0</precond-option>
          <output>10</output>
          <keep-rhs>false</keep-rhs>
      </linear-solver>

    
  </alien-bench>
</case>
