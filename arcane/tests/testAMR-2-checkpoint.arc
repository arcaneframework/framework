<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test AMR 2</titre>
  <description>Test AMR 2</description>
  <boucle-en-temps>AMRTestLoop</boucle-en-temps>
  <modules>
    <module name="ArcaneCheckpoint" active="true" />
  </modules>
 </arcane>
 
 <maillage amr="true">
  <meshgenerator><sod><x>4</x><y>1</y><z>2</z></sod></meshgenerator> 
 </maillage>
 
 <a-m-r-test>
   <format-service name="Ensight7PostProcessor">
    <fichier-binaire>false</fichier-binaire>
  </format-service>
 </a-m-r-test>
 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
