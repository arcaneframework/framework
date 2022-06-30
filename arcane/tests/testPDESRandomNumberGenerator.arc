<?xml version="1.0" encoding="ISO-8859-1"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">

  <arcane>
    <title>Test Arcane 1</title>
    <description>Test Arcane 1</description>
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
    <test name="PDESRandomNumberGeneratorUnitTest">
      <PDESRandomNumberGenerator name="PDESRandomNumberGenerator">
        <initialSeed>1234</initialSeed>
      </PDESRandomNumberGenerator>
    </test>
  </unit-test-module>
</case>