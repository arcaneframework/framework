<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test Utils 1</title>
    <description>Test Utils 1</description>
    <timeloop>UnitTest</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Cartesian2D">
        <origin>0.0 0.0</origin>
        <nb-part-x>4</nb-part-x>
        <nb-part-y>2</nb-part-y>
        <x><n>20</n><length>1.0</length></x>
        <y><n>20</n><length>1.0</length></y>
      </generator>
    </mesh>
    <mesh>
      <generator name="Cartesian2D">
        <origin>0.0 0.0</origin>
        <nb-part-x>2</nb-part-x>
        <nb-part-y>1</nb-part-y>
        <x><n>10</n><length>1.0</length></x>
        <y><n>10</n><length>1.0</length></y>
      </generator>
      <mesh-builder-customiser name="ArcaneParallelMngSplitterMeshBuilderCustomiser">
        <subset-ratio>4</subset-ratio>
      </mesh-builder-customiser>
    </mesh>
  </meshes>

  <unit-test-module>
    <test name="MeshBuilderCustomiserUnitTest"/>
  </unit-test-module>
</case>
