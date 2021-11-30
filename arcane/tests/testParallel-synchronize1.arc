<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Parallel</titre>
  <description>Test Parallel</description>
  <boucle-en-temps>TestParallel</boucle-en-temps>
 </arcane>

 <maillage>
  <meshgenerator><sod><x>100</x><y>10</y><z>10</z></sod></meshgenerator>
  <initialisation />
 </maillage>
 <parallel-tester>
  <test-id>None</test-id>
  <nb-test-sync>5</nb-test-sync>
 </parallel-tester>
</cas>
