<?xml version='1.0' encoding='ISO-8859-1'?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
	<arcane>
		<titre>Test Maillage 1</titre>
		<description>Test Maillage 1</description>
		<boucle-en-temps>UnitTest</boucle-en-temps>
	</arcane>
	<maillage>
		<fichier internal-partition="true">sphere.vtk</fichier>
	</maillage>

	<module-test-unitaire>
		<test name="GeometryUnitTest">
			<use-external-storage>true</use-external-storage>
			<factor>0.5</factor>
			<niter>4</niter>
			<geometry name="Euclidian3Geometry" />
		</test>
	</module-test-unitaire>
</cas>
