<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <titre>Test MeshCriteriaLoadBalanceMng</titre>
    <description>Test MeshCriteriaLoadBalanceMng</description>
    <timeloop>MeshCriteriaLoadBalanceMngTestModuleLoop</timeloop>
    <modules>
      <module name="MeshCriteriaLoadBalanceMngTest" active="true"/>
    </modules>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Cartesian2D" >
        <nb-part-x>4</nb-part-x>
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
        <nb-part-y>2</nb-part-y>
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
  </meshes>

  <mesh-criteria-load-balance-mng-test>
  </mesh-criteria-load-balance-mng-test>

</case>
