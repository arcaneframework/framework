<?xml version='1.0' encoding='ISO-8859-1'?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
	<arcane>
		<titre>Test Solver</titre>
		<description>Test interface solver</description>
		<boucle-en-temps>UnitTest</boucle-en-temps>
	</arcane>
	<maillage>
		<fichier internal-partition="true">sphere.vtk</fichier>
	</maillage>

	<module-test-unitaire>
		<test name="SolverUnitTest">
			<trans>2.</trans>
			<bc-trans>2.</bc-trans>
			<divKGradScheme name="DivKGradFiniteVolumeScheme" />
			<linearsolver name="HypreSolver">
				<solver>BiCGStab</solver>
				<preconditioner>None</preconditioner>
				<verbose>true</verbose>
				<num-iterations-max>40</num-iterations-max>
				<stop-criteria-value>1e-6</stop-criteria-value>
			</linearsolver>
		</test>
	</module-test-unitaire>
</cas>
