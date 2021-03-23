<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Particule 1</titre>
  <description>Test de la gestion des particules</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <meshgenerator><sod><x>100</x><y>5</y><z>5</z></sod></meshgenerator>
 </maillage>

 <module-test-unitaire>
  <test name="ParticleUnitTest">
   <max-iteration>10</max-iteration>
   <nb-particule-par-maille>2</nb-particule-par-maille>
  </test>
 </module-test-unitaire>
</cas>
