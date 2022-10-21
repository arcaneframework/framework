<?xml version="1.0"?>
<case codename="stc" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Examples STC</title>
    <timeloop>example2</timeloop>
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

  <!-- //! [SimpleTableComparatorExample2_arc]  -->
  <simple-table-comparator-example2>
    <st-output name="SimpleCsvOutput">
      <tableDir>example2</tableDir>
      <tableName>Results_Example2</tableName>
    </st-output>

    <st-comparator name="SimpleCsvComparator">
    </st-comparator>
  </simple-table-comparator-example2>
  <!-- //! [SimpleTableComparatorExample2_arc]  -->

</case>
