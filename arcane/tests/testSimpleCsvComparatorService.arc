﻿<?xml version="1.0" encoding="UTF-8"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test Service SimpleCsvComparator 1</title>
    <timeloop>UnitTest</timeloop>
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


  <unit-test-module>

    <xml-test name="SimpleTableComparatorUnitTest">
      <simple-table-comparator name="SimpleCsvComparator">
      </simple-table-comparator>
      <simple-table-output name="SimpleCsvOutput">
      </simple-table-output>
    </xml-test>

  </unit-test-module>

</case>