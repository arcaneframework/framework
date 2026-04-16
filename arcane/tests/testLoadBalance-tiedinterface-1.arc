<?xml version='1.0'?>
<cas codeversion='1.0' codename='ArcaneTest' xml:lang='fr'>
  <arcane>
    <titre>Test Equilibrage 1</titre>
    <description>Test des routines d'equilibrage de charge</description>
    <boucle-en-temps>TestParallel</boucle-en-temps>
  </arcane>

  <maillage utilise-unite="0">
    <fichier cut_4='soudures_3_cut_4'>soudures_3.mli</fichier>
    <interfaces-liees>
      <semi-conforme esclave="INTERFACE1" />
    </interfaces-liees>
    <initialisation />
  </maillage>
  <parallel-tester>
    <test-id>TestAll</test-id>
    <service-equilibrage-charge name="DefaultPartitioner" />
  </parallel-tester>
</cas>
