<?xml version="1.0" ?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test Maillage 1</titre>
    <description>Test Maillage 1</description>
    <boucle-en-temps>UnitTest</boucle-en-temps>
  </arcane>

  <meshes>
    <mesh>
      <!-- <maillage nb-ghostlayer='0' ghostlayer-builder-version="3"> -->
      <filename>mesh_with_loose_items.msh</filename>
      <nb-ghost-layer>0</nb-ghost-layer>
      <cell-dimension-kind>non-manifold</cell-dimension-kind>
    </mesh>
  </meshes>

  <module-test-unitaire>
    <test name="MeshUnitTest">
      <compute-owners-direct>true</compute-owners-direct>
      <test-adjacence>false</test-adjacence>
      <ecrire-maillage>false</ecrire-maillage>
    </test>
  </module-test-unitaire>

</cas>
