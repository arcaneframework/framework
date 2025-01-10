<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0" codeunit="CGS">
 <arcane>
  <titre>Test Maillage 1</titre>
  <description>Test Maillage 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage utilise-unite="true">
  <fichier internal-partition="true">tube5x5x100.mli2</fichier>
 </maillage>

 <module-test-unitaire>
  <test name="MeshUnitTest">
   <maillage-additionnel>tube5x5x100.mli2</maillage-additionnel>
 </test>
</module-test-unitaire>

</cas>
