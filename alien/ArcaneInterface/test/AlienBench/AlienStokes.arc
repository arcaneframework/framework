<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="AlienBench" xml:lang="en">
  <arcane>
    <title>Test unitaires des solveurs lineaires</title>
    <timeloop>StokesLoop</timeloop>
  </arcane>

  <arcane-post-processing>
    <output-period>1</output-period>
    <output>
      <variable>V</variable>
      <variable>P</variable>
      <variable>E</variable>
      <variable>XV</variable>
      <variable>XP</variable>
      <variable>XE</variable>
    </output>
  </arcane-post-processing>

  <mesh>
    <meshgenerator>
      <cartesian>
        <origine>0 0 0</origine>
        <nsd>1 1 1</nsd>
        <lx nx="4">1.</lx>
        <ly ny="4">1.</ly>
        <lz nz="4">1.</lz>
      </cartesian>
    </meshgenerator>
  </mesh>


    <alien-stokes>

      <uzawa-max-nb-iterations>3</uzawa-max-nb-iterations>
      <uzawa-factor>0.5</uzawa-factor>
      <linear-solver name="PETScSolver">
        <solver name="BiCGStab">
          <num-iterations-max>1000</num-iterations-max>
          <stop-criteria-value>1e-8</stop-criteria-value>
          <preconditioner name="BlockILU">
          </preconditioner>
        </solver>

       <verbose>high</verbose>
      </linear-solver>
      <!--linear-solver name="PETScSolver">
        <solver name="SuperLU"/>
      </linear-solver-->
      <!--linear-solver name="IFPSolver">
          <num-iterations-max>1000</num-iterations-max>
          <stop-criteria-value>1e-8</stop-criteria-value>
          <precond-option>ILU0</precond-option>
          <output>10</output>
          <keep-rhs>false</keep-rhs>
      </linear-solver-->
    
  </alien-stokes>
</case>
