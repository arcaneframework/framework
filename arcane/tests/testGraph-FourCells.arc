<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Graph 1</titre>
  <description>Test Graph 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
   <!--  fichier internal-partition="true">sphere.vtk</fichier-->
   <meshgenerator>
      <cartesian>
        <origine>0 0 0</origine>
        <nsd>4 1 1</nsd>
        <lx nx="4">4.</lx>
        <ly ny="1">1.</ly>
        <lz nz="1">1.</lz>
      </cartesian>
    </meshgenerator>   
   <partitionneur>Titi</partitionneur>
 </maillage>

 <module-test-unitaire>
  <test name="GraphUnitTest" />
 </module-test-unitaire>

</cas>
