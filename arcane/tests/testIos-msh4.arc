<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test IOS Reader/Writer MSH</titre>
  <description>Lecture/Ecriture d'un fichier au format MSH</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <fichier internal-partition='true'>sod3d-misc.msh</fichier>
 </maillage>

 <module-test-unitaire>
  <test name="IosUnitTest">
   <ecriture-vtu>false</ecriture-vtu>
   <ecriture-xmf>false</ecriture-xmf>
   <ecriture-msh>false</ecriture-msh>
  </test>
 </module-test-unitaire>

</cas>
