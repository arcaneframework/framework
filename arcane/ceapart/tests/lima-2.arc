<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage 1</titre>
  <description>Test Maillage 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage use-unit="false">
  <fichier internal-partition="true">tube5x5x100.unf</fichier>
 </maillage>

 <module-test-unitaire>
  <test name="MeshUnitTest">
   <maillage-additionnel>tube5x5x100.mli2</maillage-additionnel>
   <maillage-additionnel>tube5x5x100.unf</maillage-additionnel>
  </test>
 </module-test-unitaire>

</cas>
