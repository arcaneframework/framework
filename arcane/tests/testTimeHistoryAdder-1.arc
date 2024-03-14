<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test 1</title>
    <timeloop>TimeHistoryAdderTestModuleLoop</timeloop>
    <modules>
      <module name="TimeHistoryAdderTest" active="true" />
    </modules>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Cartesian2D" >
        <nb-part-x>2</nb-part-x>
        <nb-part-y>2</nb-part-y>
        <origin>0.0 0.0</origin>
        <x><n>4</n><length>4.0</length></x>
        <y><n>4</n><length>4.0</length></y>
      </generator>
    </mesh>
    <mesh>
      <generator name="Cartesian2D" >
        <nb-part-x>2</nb-part-x>
        <nb-part-y>2</nb-part-y>
        <origin>0.0 0.0</origin>
        <x><n>4</n><length>4.0</length></x>
        <y><n>4</n><length>4.0</length></y>
      </generator>
    </mesh>
  </meshes>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter">
      <format-version>3</format-version>
    </checkpoint-service>
    <do-dump-at-end>true</do-dump-at-end>
  </arcane-checkpoint>

  <time-history-mng-test>

  </time-history-mng-test>


</case>
