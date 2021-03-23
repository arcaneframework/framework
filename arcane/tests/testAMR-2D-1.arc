<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test AMR 2</titre>
  <description>Test AMR 2</description>
  <boucle-en-temps>AMRTestLoop</boucle-en-temps>
  <modules>
  </modules>
 </arcane>
 
 <maillage amr="true">
  <meshgenerator><sod><x>8</x><y>6</y></sod></meshgenerator> 
 </maillage>
 
 <a-m-r-test>
   <format-service name="Ensight7PostProcessor">
    <fichier-binaire>false</fichier-binaire>
   </format-service>
   <amr-ratio>0.6</amr-ratio>
 </a-m-r-test>
</cas>
