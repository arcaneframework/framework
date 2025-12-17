<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Arcane 1</titre>
  <description>Test Arcane 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
  <modules>
   <module name="ArcanePostProcessing" active="true" />
  </modules>
 </arcane>

    <meshes>
        <mesh>
            <filename>faultx1_2x1x1.vtk</filename>
            <specific-reader name="VtkPolyhedralCaseMeshReader">
                <print-mesh-infos>true</print-mesh-infos>
                <print-debug-infos>false</print-debug-infos>
            </specific-reader>
        </mesh>
    </meshes>

 <module-test-unitaire>
  <test name="ExchangeItemsUnitTest">
    <test-operation>repartition-cells</test-operation>
  </test>
 </module-test-unitaire>

 <arcane-post-traitement>
   <periode-sortie>1</periode-sortie>
   <!-- <format name="EnsightHdfPostProcessor" /> -->
   <sortie-fin-execution>true</sortie-fin-execution>
   <sauvegarde-initiale>true</sauvegarde-initiale>
  <depouillement>
    <variable>CellFamilyNewOwnerName</variable>
    <variable>ExchangeItemsTest_CellUids</variable>
    <variable>GhostPP</variable>
    <variable>NodeGhostPP</variable>
    <variable>FaceGhostPP</variable>
    <groupe>AllCells</groupe>
<!--    <groupe>AllNodes</groupe>-->
<!--    <groupe>AllFaces</groupe>-->
  </depouillement>
  <format-service name="Ensight7PostProcessor">
   <fichier-binaire>false</fichier-binaire>
  </format-service>

 </arcane-post-traitement>

</cas>
