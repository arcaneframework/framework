<?xml version="1.0" ?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test Maillage 1</titre>
    <description>Test Maillage 1</description>
    <boucle-en-temps>UnitTest</boucle-en-temps>
  </arcane>

  <maillage>
    <fichier internal-partition="true">tube2x2x4.vtk</fichier>
  </maillage>

  <module-test-unitaire>
    <test name="MeshUnitTest">
    </test>
  </module-test-unitaire>

</cas>
