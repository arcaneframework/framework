<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <titre>Test MeshCriteriaLoadBalanceMng</titre>
    <description>Test MeshCriteriaLoadBalanceMng avec PTScotch et Zoltan (Variant 1 / 4 proc)</description>
    <timeloop>MeshCriteriaLoadBalanceMngTestModuleLoop</timeloop>
    <modules>
      <module name="MeshCriteriaLoadBalanceMngTest" active="true"/>
    </modules>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Cartesian2D" >
        <nb-part-x>2</nb-part-x>
        <nb-part-y>2</nb-part-y>
        <origin>0.0 0.0</origin>
        <x>
          <n>5</n>
          <length>5.0</length>
        </x>
        <y>
          <n>3</n>
          <length>3.0</length>
        </y>
      </generator>
    </mesh>

    <mesh>
      <generator name="Cartesian3D">
        <nb-part-x>2</nb-part-x>
        <nb-part-y>1</nb-part-y>
        <nb-part-z>2</nb-part-z>
        <origin>0.0 0.0 0.0</origin>
        <x>
          <n>5</n>
          <length>5.0</length>
        </x>
        <y>
          <n>5</n>
          <length>5.0</length>
        </y>
        <z>
          <n>4</n>
          <length>4.0</length>
        </z>
      </generator>
    </mesh>

    <mesh>
      <generator name="Sod3D">
        <x>50</x>
        <y>5</y>
        <z>5</z>
      </generator>
    </mesh>

    <mesh>
      <filename internal-partition="true">sphere_v4.2.vtk</filename>
    </mesh>
  </meshes>

  <mesh-criteria-load-balance-mng-test>
    <mesh-params>
      <partitioner>PTScotch</partitioner>
      <iteration>1</iteration>
    </mesh-params>
    <mesh-params>
      <partitioner>Zoltan</partitioner>
      <iteration>2</iteration>
    </mesh-params>
    <mesh-params>
      <partitioner>PTScotch</partitioner>
      <iteration>2</iteration>
    </mesh-params>
    <mesh-params>
      <partitioner>Zoltan</partitioner>
      <iteration>1</iteration>
    </mesh-params>
  </mesh-criteria-load-balance-mng-test>

</case>
