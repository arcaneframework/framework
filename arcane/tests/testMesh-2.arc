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
    <connectivity-file-checksum>94fd015ce65dbb5f87d7b221c8e0a2c73a734f25e265ff474173e6e765c346db</connectivity-file-checksum>
    <connectivity-file-checksum-parallel>f180ff21d9823c7a650059749e33aac1eed9390a3b7ce2fd40700b4e9c9cdfb6</connectivity-file-checksum-parallel>
    <connectivity-file-checksum-parallel>dc9e37707d0d9dab9338c63aedd1840e506859eea1ca45f4e2e50f79d3aa8fcb</connectivity-file-checksum-parallel>
    <connectivity-file-checksum-parallel>92a4bcda45163ac93f0510b73b10987a91bd9ed855e6a96af0ba3d1154fb3e3f</connectivity-file-checksum-parallel>
    <connectivity-file-checksum-parallel>1eda7a4d1dd3685f30e577f010f51c6423e9b4d45ae972cd21453acad7b97362</connectivity-file-checksum-parallel>
  </test>
 </module-test-unitaire>

</cas>
