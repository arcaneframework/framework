<?xml version='1.0'?>
<cas codeversion='1.0' codename='ArcaneTest' xml:lang='fr'>
 <arcane>
  <titre>Test Equilibrage 1</titre>
  <description>Test des routines equilibrage de charge</description>
  <boucle-en-temps>TestParallel</boucle-en-temps>
  <!-- <boucle-en-temps>TestLoadBalancing</boucle-en-temps> -->
 </arcane>

 <maillage>
  <fichier internal-partition="true">sod.vtk</fichier>
  <initialisation />
 </maillage>
 <parallel-tester>
  <test-id>TestAll</test-id>
  <service-equilibrage-charge name="DefaultPartitioner" />
 </parallel-tester>
</cas>
