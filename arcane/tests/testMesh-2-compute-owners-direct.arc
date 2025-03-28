<?xml version="1.0" ?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test Maillage 1</titre>
    <description>Test Maillage 1</description>
    <boucle-en-temps>UnitTest</boucle-en-temps>
  </arcane>

  <maillage nb-ghostlayer='0' ghostlayer-builder-version="3">
    <fichier internal-partition="true">sod.vtk</fichier>
  </maillage>

  <module-test-unitaire>
    <test name="MeshUnitTest">
      <compute-owners-direct>true</compute-owners-direct>
    </test>
  </module-test-unitaire>

</cas>
