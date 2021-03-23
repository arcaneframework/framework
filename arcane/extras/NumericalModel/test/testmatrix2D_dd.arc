<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="NumericalModelTest" xml:lang="en">
  <arcane>
    <title>Experimentation Arcane</title>
    <timeloop>TimeLoop</timeloop>
  </arcane>

  <arcane-post-processing>
    <output-period>1</output-period>
    <output>
      <variable>UConcentration</variable>
      <variable>VConcentration</variable>
      <variable>CellVelocity</variable>
      <variable>CellRValue</variable>
    </output>
  </arcane-post-processing>

  <mesh>
    <file internal-partition='true'>Mesh2DCoarse.vt2</file>
    <initialisation>
      <variable nom="UConcentration" valeur="0." groupe="AllCells" />
      <variable nom="VConcentration" valeur="0." groupe="AllCells" />
      <variable nom="Permeability" valeur="100e" groupe="AllCells" />
      <variable nom="Rho" valeur="1." groupe="AllCells" />
      <variable nom="Compressibility" valeur="1." groupe="AllCells" />
      <variable nom="CellPermx" valeur="0e+6" groupe="AllCells" />
      <variable nom="CellPermy" valeur="0e+6" groupe="AllCells" />
      <variable nom="CellPermz" valeur="0e+6" groupe="AllCells" />
      <variable nom="Velocity" valeur="1.0" groupe="AllFaces" />
    </initialisation>
  </mesh>

  <output-level>10</output-level>

  <toy-reactive-transport-couplage-dt-local>
    <output-level>10</output-level>

    <DomainGlobal name="ToyReactiveTransportModel">
      <output-level>13</output-level>
      <diffSchemeFluxDomain name="DivKGradTwoPoints" />
      <advSchemeFluxDomain name="AdvectionUpwind" />
      <interpolator name="GaussInterpolator" />
      <linearsolver name="HypreSolver">
        <num-iterations-max>40</num-iterations-max>
        <stop-criteria-value>1e-6</stop-criteria-value>
        <solver>GMRES</solver>
        <preconditioner>Euclid</preconditioner>
        <verbose>true</verbose>
      </linearsolver>
      <time-step-mng name="VarTimeStepMng">
        <type>Geometric</type>
        <increase-factor>2.0</increase-factor>
        <decrease-factor>0.5</decrease-factor>
      </time-step-mng>
      <!--
        =====================================================================
        RHS, k, Psi
        =====================================================================
      -->
      <rhs>0</rhs>
      <k>100*h(2250-x)*h(x-750)*h(y-1250)</k>
      <psi>u</psi>
      <flow-model name="BasicFlowModel">
        <linear-solver-flow name="HypreSolver">
          <num-iterations-max>200</num-iterations-max>
          <stop-criteria-value>1e-6</stop-criteria-value>
          <solver>GMRES</solver>
          <preconditioner>AMG</preconditioner>
          <verbose>true</verbose>
        </linear-solver-flow>
        
        <opflow name="DivKGradTwoPoints" />
        
        <interpolator name="GaussInterpolator" />

        <flux-term-service name="FluxTermMng" />

        <flow-boundary-condition-mng name="BoundaryConditionMng">

          <bc>
            <tag>DefaultFaces</tag>
            <type>Neumann</type>
            <value>"0."</value>
          </bc>

          <bc>
            <tag>Injector1Faces</tag>
            <type>Dirichlet</type>
            <value>"1100000."</value>
          </bc>

          <bc>
            <tag>Injector2Faces</tag>
            <type>Dirichlet</type>
            <value>"1050000."</value>
          </bc>

          <bc>
            <tag>ProductorFaces</tag>
            <type>Dirichlet</type>
            <value>"100."</value>
          </bc>

        </flow-boundary-condition-mng>

      </flow-model>

    </DomainGlobal>

    <DomainWindow name="ToyReactiveTransportModel">
      <output-level>13</output-level>
      <diffSchemeFluxDomain name="DivKGradTwoPoints" />
      <advSchemeFluxDomain name="AdvectionUpwind" />
      <interpolator name="GaussInterpolator" />
      <linearsolver name="HypreSolver">
        <num-iterations-max>40</num-iterations-max>
        <stop-criteria-value>1e-6</stop-criteria-value>
        <solver>GMRES</solver>
        <preconditioner>Euclid</preconditioner>
        <verbose>true</verbose>
      </linearsolver>
      <time-step-mng name="VarTimeStepMng">
        <type>Geometric</type>
        <increase-factor>2.0</increase-factor>
        <decrease-factor>0.5</decrease-factor>
      </time-step-mng>
      <!--
        =====================================================================
        RHS, k, Psi
        =====================================================================
      -->
      <rhs>0</rhs>
      <k>100*h(2250-x)*h(x-750)*h(y-1250)</k>
      <!--k>0</k-->
      <psi>u</psi>

    </DomainWindow>

    <DomainReservoir name="ToyReactiveTransportModel">
      <output-level>13</output-level>
      <diffSchemeFluxDomain name="DivKGradTwoPoints" />
      <advSchemeFluxDomain name="AdvectionUpwind" />
      <interpolator name="GaussInterpolator" />
      <linearsolver name="HypreSolver">
        <num-iterations-max>40</num-iterations-max>
        <stop-criteria-value>1e-6</stop-criteria-value>
        <solver>GMRES</solver>
        <preconditioner>Euclid</preconditioner>
        <verbose>true</verbose>
      </linearsolver>
      <time-step-mng name="VarTimeStepMng">
        <type>Geometric</type>
        <increase-factor>2.0</increase-factor>
        <decrease-factor>0.5</decrease-factor>
      </time-step-mng>
      <!--
        =====================================================================
        RHS, k, Psi
        =====================================================================
      -->
      <rhs>0</rhs>
      <k>100*h(2250-x)*h(x-750)*h(y-1250)</k>
      <!--k>0</k-->
      <psi>u</psi>

    </DomainReservoir>

    <overlap-diff-flux-scheme name="DivKGradTwoPoints">
    </overlap-diff-flux-scheme>
    <overlap-adv-flux-scheme name="AdvectionUpwind">
    </overlap-adv-flux-scheme>

    <window>WINDOW</window>
    <reservoir>RESERVOIR</reservoir>

    <fine-time-mng name="TimeMng">
      <init-time>0</init-time>
      <end-time>1.</end-time>
      <init-time-step>0.005</init-time-step>
      <min-time-step>0.000000000000000000001</min-time-step>
      <max-time-step>0.0051</max-time-step>
    </fine-time-mng>

    <time-step-mng name="VarTimeStepMng">
      <type>Geometric</type>
      <increase-factor>1.0</increase-factor>
      <decrease-factor>1.0</decrease-factor>
    </time-step-mng>

    <computation-opt>DFVF1</computation-opt>
    <!--computation-opt>DFVF2</computation-opt-->
    <!--computation-opt>GlobalCoarseDt</computation-opt-->
    <!--computation-opt>GlobalFineDt</computation-opt-->

    <max-iter>20</max-iter>

    <boundary-condition>
      <type>Dirichlet</type>
      <value>1.</value>
      <surface>Injector1Faces</surface>
    </boundary-condition>
    <boundary-condition>
      <type>Dirichlet</type>
      <value>1.0</value>
      <surface>Injector2Faces</surface>
    </boundary-condition>
    <boundary-condition>
      <type>Dirichlet</type>
      <value>0.0</value>
      <surface>ProductorFaces</surface>
    </boundary-condition>

  </toy-reactive-transport-couplage-dt-local>

  <shpco2-arcane>
    <!--// Groups Definition //-->
    <group-creator name="StandardGroupCreator">
      <scalar-eval>false</scalar-eval>
      <!--// Cell Groups //-->
      <cellgroup>
        <name>FULLDOMAIN</name>
        <area>AllCells</area>
        <filter>1</filter>
      </cellgroup>
      <cellgroup>
        <name>WINDOW</name>
        <area>AllCells</area>
        <filter>h(2250-x)*h(x-750)*h(y-1250)</filter>
        <filter-test>true</filter-test>
      </cellgroup>
      <cellgroup>
        <name>RESERVOIR</name>
        <area>FULLDOMAIN</area>
        <filter>h(2250-x)*h(x-750)*h(y-1250)</filter>
        <filter-test>false</filter-test>
      </cellgroup>
      <cellgroup>
        <name>Barriers</name>
        <area>TrapCells</area>
        <filter>h(x-1000)*h(1250-x)*h(2000-y) + h(x-2250)*h(2500-x)*h(y-1000) + h(x-3500)*h(3750-x)*h(y-500)</filter>
      </cellgroup>

      <facegroup>
        <name>testallfaes</name>
        <area>AllFaces</area>
        <filter>1</filter>
      </facegroup>

      <facegroup>
        <name>DefaultFaces</name>
        <area>GC_allBoundaryFaces</area>
        <filter>1.</filter>
      </facegroup>
      <facegroup>
        <name>Injector1Faces</name>
        <area>GC_allBoundaryFaces</area>
        <filter>h(x+0.000001)*h(0.000001-x)*h(1000-y)*h(y)</filter>
      </facegroup>
      <facegroup>
        <name>Injector2Faces</name>
        <area>GC_allBoundaryFaces</area>
        <filter>h(y-3000+0.000001)*h(0.000001-y+3000)*h(x-2500)*h(3500-x)</filter>
      </facegroup>
      <facegroup>
        <name>ProductorFaces</name>
        <area>GC_allBoundaryFaces</area>
        <filter>h(x-4750+0.000001)*h(4750-x+0.000001)*h(y)*h(3000-y)</filter>
      </facegroup>
      <!--// Groups Alias for BC-Mng //-->
      <facegroup>
        <name>BC_DefaultFaces</name>
        <area>DefaultFaces</area>
        <filter>1.</filter>
      </facegroup>
      <facegroup>
        <name>BC_Injector1Faces</name>
        <area>Injector1Faces</area>
        <filter>1.</filter>
      </facegroup>
      <facegroup>
        <name>BC_Injector2Faces</name>
        <area>Injector2Faces</area>
        <filter>1.</filter>
      </facegroup>
      <facegroup>
        <name>BC_ProductorFaces</name>
        <area>ProductorFaces</area>
        <filter>1.</filter>
      </facegroup>

    </group-creator>
  
    <time-mng name="TimeMng">
      <init-time>0</init-time>
      <end-time>1.</end-time>
      <init-time-step>0.01</init-time-step>
      <min-time-step>0.000000000000000000000001</min-time-step>
      <max-time-step>0.0100000001</max-time-step>
    </time-mng>
    
    <geometry-service name="Euclidian3Geometry" />
        
    <expression-mng name="ExpressionMng" />
    
    <exp-parser name="ExpressionParser" />
    
    <validator name="SyntheticServiceValidator">
      <variable-field name="VariableAccessor">
        <name>UConcentration</name>
      </variable-field>
      <group-name>AllCells</group-name>
      <reference-value>0.9543953315</reference-value>
      <reduction>Mean</reduction>
      <comparator>AbsoluteError</comparator>
      <tolerance>1e-7</tolerance>
      <verbose>true</verbose>
    </validator>
    <validator name="SyntheticServiceValidator">
      <variable-field name="VariableAccessor">
        <name>VConcentration</name>
      </variable-field>
      <group-name>AllCells</group-name>
      <reference-value>0.1761699293</reference-value>
      <reduction>Mean</reduction>
      <comparator>AbsoluteError</comparator>
      <tolerance>1e-7</tolerance>
      <verbose>true</verbose>
    </validator>
  
  </shpco2-arcane>
</case>
