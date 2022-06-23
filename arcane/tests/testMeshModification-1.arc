<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Arcane 1</titre>
  <description>Test Arcane 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
  <modules>
   <module name="ArcanePostProcessing" active="true" />
  </modules>
 </arcane>

 <maillage>
  <meshgenerator><sod><x>10</x><y>2</y><z>2</z></sod></meshgenerator>
 </maillage>

 <module-test-unitaire>
   <test name="MeshModification">
     <remove-fraction>2</remove-fraction>
   </test>
 </module-test-unitaire>

 <arcane-post-traitement>
   <periode-sortie>1</periode-sortie>
   <!-- <format name="EnsightHdfPostProcessor" /> -->
   <sortie-fin-execution>true</sortie-fin-execution>
  <depouillement>
    <variable>CellFamilyNewOwnerName</variable>
    <groupe>AllCells</groupe>
  </depouillement>
  <format-service name="Ensight7PostProcessor">
   <fichier-binaire>false</fichier-binaire>
  </format-service>

 </arcane-post-traitement>

</cas>
