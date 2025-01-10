<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0" xml:lang="en">
  <arcane>
    <title>Test subdivision d'un maillage 3D hexahédrique</title>
    <description>Subdivision d'un maillage 3D uniquement hexahédrique avec le partitionneur par défaut</description>
    <timeloop>UnitTest</timeloop>
  </arcane>

  <meshes>
    <mesh>
<!--        <filename>one_tet.msh</filename>-->
     <!-- <filename>sod.vtk</filename> -->
     <!-- <filename>subdivider_one_hexa_ouput.vtk</filename> -->
         <!-- <generator name="Cartesian3D" >
        <nb-part-x>1</nb-part-x>
        <nb-part-y>0</nb-part-y>
        <nb-part-z>0</nb-part-z>
        <origin>0.0 0.0 0.0</origin>

            

        <x><n>1</n><length>4.0</length></x>
        <y><n>1</n><length>4.0</length></y>
        <z><n>1</n><length>4.0</length></z>
        <face-numbering-version>1</face-numbering-version> -->
        
            <!-- <x><n>2</n><length>2.0</length><progression>1.0</progression></x>
            <x><n>3</n><length>3.0</length><progression>4.0</progression></x>
            <x><n>3</n><length>3.0</length><progression>8.0</progression></x>
            <x><n>4</n><length>4.0</length><progression>16.0</progression></x>

            <y><n>2</n><length>2.0</length><progression>1.0</progression></y>
            <y><n>3</n><length>3.0</length><progression>4.0</progression></y>
            <y><n>3</n><length>3.0</length><progression>8.0</progression></y>

            <z><n>2</n><length>2.0</length><progression>1.0</progression></z>
            <z><n>3</n><length>3.0</length><progression>2.0</progression></z>
            <z><n>3</n><length>4.0</length><progression>3.0</progression></z> -->

        <!-- </generator> -->
        
        <!-- 2D -->
        <generator name="Cartesian2D" >
            <nb-part-x>2</nb-part-x>
            <nb-part-y>0</nb-part-y>
            <origin>0.0 0.0</origin>
            <x><n>10</n><length>4.0</length></x>
            <y><n>10</n><length>4.0</length></y>
            <face-numbering-version>1</face-numbering-version>
        </generator>
      <!-- <filename>Cow.msh</filename> -->
      <!-- <filename>Indorelax.msh</filename> -->
      <partitioner>MeshPartitionerTester</partitioner>
      <subdivider>
          
          <nb-subdivision>1</nb-subdivision>
          
      </subdivider>
    </mesh>
  </meshes>
  <unit-test-module>
    <test name="MeshUnitTest">
      <test-adjency>0</test-adjency>
    </test>
  </unit-test-module>

</case>
