<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Soudure multiples sur une meme maille</titre>
  <description>Test Soudure multiples sur une meme maille</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage utilise-unite="0">
  <fichier cut_4='sphere_multi_soudure_cut_4'>sphere_multi_soudure.unf</fichier>
 </maillage>
 <module-test-unitaire>
  <test name="MeshUnitTest">
   <test-ecrivain-variable>true</test-ecrivain-variable>
  </test>
 </module-test-unitaire>
</cas>
