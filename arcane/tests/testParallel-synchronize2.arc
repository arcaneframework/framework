<?xml version="1.0" ?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test Parallel</titre>
    <description>Test Parallel</description>
    <boucle-en-temps>TestParallel</boucle-en-temps>
  </arcane>

  <meshes>
    <mesh>
      <ghost-layer-builder-version>4</ghost-layer-builder-version>
      <generator name="Cartesian3D" >
        <nb-part-x>2</nb-part-x>
        <nb-part-y>2</nb-part-y>
        <nb-part-z>2</nb-part-z>
        <origin>0.0 0.0 0.0</origin>
        <x><n>100</n><length>1.0</length></x>
        <y><n>10</n><length>1.0</length></y>
        <z><n>40</n><length>1.0</length></z>
        <face-numbering-version>1</face-numbering-version>
      </generator>
    </mesh>
  </meshes>

  <parallel-tester>
    <test-id>None</test-id>
    <nb-test-sync>3</nb-test-sync>
  </parallel-tester>
</cas>
