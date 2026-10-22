<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0" xml:lang="en">
  <arcane>
    <title>Test subdivision d'un maillage 2D triangulaire</title>
    <description>Subdivision d'un maillage 2D uniquement triangulaire avec le partitionneur par défaut</description>
    <timeloop>MeshSubdividerTesterLoop</timeloop>
  </arcane>
  <meshes>
    <mesh>
      <filename>plancher.msh</filename>
<!--      <partitioner>MeshPartitionerTester</partitioner>-->
      <partitioner>Metis</partitioner>
      <subdivider>
          <nb-subdivision>1</nb-subdivision>
      </subdivider>
      <initialization>
      </initialization>
    </mesh>
  </meshes>

  <mesh-subdivider-tester>
<!--      <test name="MeshUnitTest">-->
<!--        <test-adjency>1</test-adjency>-->
<!--        <write-mesh>false</write-mesh>-->
<!--        <write-mesh-service-name>VtkLegacyMeshWriter</write-mesh-service-name>-->
<!--        <output-file>base_name</output-file>-->
<!--      </test>-->
      <!-- Volume -->
      <!-- UniqueID -->
  </mesh-subdivider-tester>
  <arcane-post-processing>
    <output-period>1</output-period>
    <output>
      <variable>arcane_cell_uniqueid</variable>
      <variable>arcane_cell_rank</variable>
    </output>
  </arcane-post-processing>

</case>
