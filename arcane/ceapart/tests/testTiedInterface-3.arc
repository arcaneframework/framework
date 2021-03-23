<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage 1</titre>
  <description>Test Maillage 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage utilise-unite="false">
  <fichier cut_4='soudures_3_cut_4'>soudures_3.mli</fichier>
  <interfaces-liees>
   <semi-conforme esclave="INTERFACE1" />
  </interfaces-liees>
 </maillage>
 <module-test-unitaire>
  <test name="MeshUnitTest">
   <test-ecrivain-variable>true</test-ecrivain-variable>
   <ecrire-maillage>0</ecrire-maillage>
   <test-adjacence>0</test-adjacence>
  </test>
 </module-test-unitaire>
</cas>
