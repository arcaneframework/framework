<?xml version="1.0" ?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage 1</titre>
  <description>Test Maillage 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <fichier internal-partition="true">sod.vtk</fichier>
 </maillage>

 <module-test-unitaire>
  <test name="MeshUnitTest">
    <maillage-additionnel>sod.vtk</maillage-additionnel>
    <create-edges>true</create-edges>
    <connectivity-file-checksum>655f3db783f901499101eadf4171d5cd8d2ca64f598ee7b35b75dd483572aa3d</connectivity-file-checksum>
    <!-- <connectivity-file-checksum-parallel>12b9e0d2e8fa0a70a9a61c4dddf1633ba6ab7d5947acc358a05919513200c256</connectivity-file-checksum-parallel>
    <connectivity-file-checksum-parallel>73ba56c4b1fa81a39cdd3a386cf4398ad2314ca8630443eedda1eb1bc822a252</connectivity-file-checksum-parallel>
    <connectivity-file-checksum-parallel>a83e3c9d492b0b7aec938c8fc23d3f8fc0e39199d8fb66927fcf8627e9a4c6f5</connectivity-file-checksum-parallel>
    <connectivity-file-checksum-parallel>0c6b249875ab7a75c24c1d3e608fbdd1a4e80613903b7240a1d71e8a51d89307</connectivity-file-checksum-parallel> -->
  </test>
 </module-test-unitaire>

</cas>
