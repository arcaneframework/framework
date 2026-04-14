<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Material Synchronisation 1</titre>
  <description>Test Material Synchronisation 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <meshgenerator>
   <sod><x>100</x><y>5</y><z>5</z></sod>
  </meshgenerator>
 </maillage>

 <module-test-unitaire>
  <test name="MeshMaterialSyncUnitTest">
   <nb-material>8</nb-material>
  </test>
 </module-test-unitaire>
</cas>
