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
        <nsd>2 1 1</nsd>
        <lx nx="10">1.</lx>
        <ly ny="10">1.</ly>
        <lz nz="10">1.</lz>
      </cartesian>
    </meshgenerator>
  </mesh>


    <alien-bench>
      <!-- big diagonal-coefficient keep diagonal dominant matrix -->
      <!--homogeneous>true</homogeneous-->
      <diagonal-coefficient>0.</diagonal-coefficient>
      <lambdax>0.125</lambdax>
      <lambday>0.25</lambday>
      <alpha>10.</alpha>
      <sigma>1000000.</sigma>
      <epsilon>0.01</epsilon>

      <linear-solver name="HTSSolver">
        <solver>BiCGStab</solver>
        <max-iteration-num>1000</max-iteration-num>
        <stop-criteria-value>1e-8</stop-criteria-value>
        <preconditioner>DDML</preconditioner>
        <ml-opt> 
          <algo>1</algo>
          <iter>2</iter>
          <solver>1</solver>
          <coarse-solver>0</coarse-solver>
          <coarse-op>2</coarse-op>
          <nev>5</nev>
          <evtype>2</evtype>
          <evtol>1.e-6</evtol>
        </ml-opt>
        <nb-part>2</nb-part>
        <nb-subpart>0</nb-subpart>
        <metis>1</metis>
        <smetis>1</smetis>
        <output>1</output>
      </linear-solver>

    
  </alien-bench>
</case>
