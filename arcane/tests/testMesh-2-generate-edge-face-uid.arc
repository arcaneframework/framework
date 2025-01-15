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
    <connectivity-file-checksum>67f28f6bbcc1741687bcf913688dc9067fadc22369dbb1a78f70ba45b30c0dfa</connectivity-file-checksum>
    <connectivity-file-checksum-parallel>a58676121852d7c0a8e4494fa4679e3c04b349174c1663611987b09c5ccbb393</connectivity-file-checksum-parallel>
    <connectivity-file-checksum-parallel>e3a0b58c733bfc6ae1886a7e884d480e6ad1237d9b65352f6569a626ac64f42c</connectivity-file-checksum-parallel>
    <connectivity-file-checksum-parallel>e0503600a6ea913b3a35f45859913b41c541da65fca5e7f9be47b11766b45231</connectivity-file-checksum-parallel>
    <connectivity-file-checksum-parallel>63a5ccbdba612d210ea996821cdd4fe617686bb507383de14408a7a6629ab248</connectivity-file-checksum-parallel>
  </test>
 </module-test-unitaire>

</cas>
