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
      <!-- big diagonal-coefficient keep diagonal dominant matrix -->
      <redistribution>false</redistribution>
      <diagonal-coefficient>0.</diagonal-coefficient>
      <lambdax>0.125</lambdax>
      <lambday>0.25</lambday>
      <alpha>10.</alpha>
      <sigma>1000000.</sigma>
      <epsilon>0.01</epsilon>

      <linear-solver name="PETScSolver">
        <exec-space>Device</exec-space>
        <memory-type>Host</memory-type>
        <solver name="BiCGStab">
          <num-iterations-max>1000</num-iterations-max>
          <stop-criteria-value>1e-8</stop-criteria-value>
           <preconditioner name="GAMG">
              <gamg-type>classicla</gamg-type>
              <gamg-threshold>0.15</gamg-threshold>
              <gamg-max-levels>25</gamg-max-levels>
              <gamg-agg-nsmooths>1</gamg-agg-nsmooths>
              <gamg-aggressive-coarsening>1</gamg-aggressive-coarsening>
              <gamg-aggressive-square-graph>1</gamg-aggressive-square-graph>
           </preconditioner>
         </solver>
       </linear-solver>

    
  </alien-bench>
</case>
