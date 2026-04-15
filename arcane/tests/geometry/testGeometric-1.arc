<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Geometric 1</titre>
  <description>Test Geometric 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <meshgenerator>
  <simple><mode>3</mode></simple>
<!--    <sod>
      <x set='false' delta='0.02'>20</x>
      <y set='false' delta='0.02'>5</y>
      <z set='false' delta='0.02'>5</z>
  </sod> -->
  </meshgenerator>
 </maillage>

 <module-test-unitaire>
  <test name="GeometricUnitTest" />
 </module-test-unitaire>
</cas>
