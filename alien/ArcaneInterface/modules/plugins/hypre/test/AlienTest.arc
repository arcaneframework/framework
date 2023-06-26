<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="AlienTest" xml:lang="en">
    <arcane>
        <title>Test unitaires des solveurs lineaires</title>
        <timeloop>TestLoop</timeloop>
    </arcane>

    <arcane-post-processing>
        <output-period>1</output-period>
        <output>
            <variable>U</variable>
        </output>
    </arcane-post-processing>

    <mesh>
        <!-- file internal-partition='true'>cube3D.vt2</file-->
        <file internal-partition='true'>tube5x5x100.vtk</file>
        <!--file internal-partition='true'>tube2x2x4.vtk</file-->
    </mesh>


    <alien-test>
        <!-- big diagonal-coefficient keep diagonal dominant matrix -->
        <diagonal-coefficient>24</diagonal-coefficient>
        <stencil-by>node</stencil-by>
        <vect-size>1</vect-size>

        <check-memory>false</check-memory>
        <building-only>false</building-only>

        <repeat-loop>4</repeat-loop>
        <extra-equation-count>10</extra-equation-count>

        <builder>ProfiledBuilder</builder>
        <builder>DirectBuilder</builder>

        <linear-solver name="PETScSolver">
            <solver name="BiCGStab">
                <num-iterations-max>1000</num-iterations-max>
                <stop-criteria-value>1e-8</stop-criteria-value>
                <preconditioner name="BlockILU">
                </preconditioner>
            </solver>
            <verbose>high</verbose>
        </linear-solver>

        <linear-solver name="PETScSolver">
            <solver name="BiCGStab">
                <num-iterations-max>1000</num-iterations-max>
                <stop-criteria-value>1e-8</stop-criteria-value>
                <preconditioner name="Diagonal">
                </preconditioner>
            </solver>
            <verbose>high</verbose>
        </linear-solver>


        <linear-solver name="PETScSolver">
            <solver name="GMRES">
                <num-iterations-max>1000</num-iterations-max>
                <stop-criteria-value>1e-8</stop-criteria-value>
                <preconditioner name="Hypre">
                    <field-split-mode>true</field-split-mode>
                    <type>AMG</type>
                </preconditioner>
            </solver>
        </linear-solver>

        <linear-solver name="PETScSolver">
            <solver name="BiCGStab">
                <num-iterations-max>1000</num-iterations-max>
                <stop-criteria-value>1e-8</stop-criteria-value>
                <preconditioner name="FieldSplit">
                    <verbose>true</verbose>
                    <type name="Multiplicative"/>
                    <block>
                        <tag>Elliptic</tag>
                        <solver name="Richardson">
                            <init-type>Zero</init-type>
                            <stop-criteria-value>1e-5</stop-criteria-value>
                            <preconditioner name="Hypre">
                                <field-split-mode>true</field-split-mode>
                                <type>AMG</type>
                            </preconditioner>
                            <num-iterations-max>3</num-iterations-max>
                        </solver>
                    </block>
                    <block>
                        <tag>P</tag>
                        <solver name="Richardson">
                            <init-type>Zero</init-type>
                            <stop-criteria-value>1e-5</stop-criteria-value>
                            <preconditioner name="BlockILU"/>
                            <num-iterations-max>3</num-iterations-max>
                        </solver>
                    </block>
                    <!--block>
                      <tag>Q</tag>
                      <solver name="Richardson">
                        <init-type>Zero</init-type>
                        <stop-criteria-value>1e-5</stop-criteria-value>
                        <preconditioner name="BlockILU"/>
                        <num-iterations-max>3</num-iterations-max>
                      </solver>
                    </block-->
                    <default-block>
                        <solver name="Richardson">
                            <init-type>Zero</init-type>
                            <stop-criteria-value>1e-5</stop-criteria-value>
                            <preconditioner name="BlockILU"/>
                            <num-iterations-max>3</num-iterations-max>
                        </solver>
                    </default-block>
                </preconditioner>
            </solver>
            <verbose>high</verbose>
        </linear-solver>

        <!--linear-solver name="IFPSolver">
          <verbose>true</verbose>
          <num-iterations-max>1000</num-iterations-max>
          <precond-option>CprAmg</precond-option>
        </linear-solver-->

        <!--linear-solver name="MTLSolver">
            <solver>BiCGStab</solver>
          <preconditioner>ILU0</preconditioner>
        </linear-solver-->

        <linear-solver name="HypreSolver">
            <num-iterations-max>100</num-iterations-max>
            <stop-criteria-value>1e-12</stop-criteria-value>
            <solver>GMRES</solver>
            <preconditioner>Euclid</preconditioner>
            <verbose>true</verbose>
        </linear-solver>

    </alien-test>
</case>
