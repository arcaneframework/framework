<?xml version="1.0"?>
<case codename="csv" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Examples CSV</title>
    <timeloop>example3</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Cartesian2D" >
        <face-numbering-version>1</face-numbering-version>

        <nb-part-x>1</nb-part-x> 
        <nb-part-y>1</nb-part-y>

        <origin>0.0 0.0</origin>

        <x>
          <length>1.0</length>
          <n>1</n>
        </x>

        <y>
          <length>1.0</length>
          <n>1</n>
        </y>

      </generator>
    </mesh>
  </meshes>

  <!-- //! [SimpleTableOutputExample3_arc]  -->
  <simple-table-output-example3>
    <st-output name="SimpleCsvOutput">

      <tableDir>example3</tableDir>
      <tableName>Results_Example3</tableName>

    </st-output>
  </simple-table-output-example3>
  <!-- //! [SimpleTableOutputExample3_arc]  -->

</case>
