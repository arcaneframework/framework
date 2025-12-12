<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test CartesianMesh 3D PatchCartesianMeshOnly Coarse Zone (Variant 2)</title>

    <description>
      Test du raffinement d'un maillage cartesian 3D avec le type d'AMR PatchCartesianMeshOnly
      puis du dé-raffinement de certaines zones (sans renumérotation)
    </description>

    <timeloop>AMRCartesianMeshTestLoop</timeloop>

    <modules>
      <module name="ArcanePostProcessing" active="true"/>
      <module name="ArcaneCheckpoint" active="true"/>
    </modules>

  </arcane>

  <arcane-post-processing>
    <output-period>1</output-period>
    <output>
      <variable>Density</variable>
      <variable>NodeDensity</variable>
      <group>AllCells</group>
      <group>AllNodes</group>
      <group>AllFacesDirection0</group>
      <group>AllFacesDirection1</group>
    </output>
  </arcane-post-processing>


  <mesh amr-type="3">
    <meshgenerator>
      <cartesian>
        <nsd>2 2 2</nsd>
        <origine>0.0 0.0 0.0</origine>
        <lx nx='10' prx='1.0'>10.0</lx>
        <ly ny='10' pry='1.0'>10.0</ly>
        <lz nz='10' prz='1.0'>10.0</lz>
      </cartesian>
    </meshgenerator>
  </mesh>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>

    <refinement-3d>
      <position>1.0 1.0 1.0</position>
      <length>8.0 8.0 8.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>3.0 3.0 3.0</position>
      <length>4.0 4.0 4.0</length>
    </refinement-3d>

    <coarse-zone-3d>
      <position>3.0 3.0 3.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>3.0 3.0 5.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>3.0 5.0 3.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>3.0 5.0 5.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>5.0 3.0 3.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>5.0 3.0 5.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>5.0 5.0 3.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>5.0 5.0 5.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>

    <coarse-zone-3d>
      <position>3.0 3.0 3.0</position>
      <length>3.0 3.0 3.0</length>
    </coarse-zone-3d>

    <!--    <expected-number-of-cells-in-patchs>1000 3880</expected-number-of-cells-in-patchs>-->
    <expected-number-of-cells-in-patchs>1000 160 216 240 288 432 432 576 768 768</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>194910d2bfdfb101b23a3ce372f4e60d</nodes-uid-hash>
    <faces-uid-hash>40759f3dc739bdf026ddafe426e9a28d</faces-uid-hash>
    <cells-uid-hash>b025a6b3ac6545c730498d9967a65511</cells-uid-hash>

    <nodes-direction-hash>441ff981462b70a1e075fd6c6d9ba4f3</nodes-direction-hash>
    <faces-direction-hash>5882e4443dcbebdcf9f2122f5d49c110</faces-direction-hash>
    <cells-direction-hash>4f219d63c8c8482f4fdf98f67b91717f</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter"/>
  </arcane-checkpoint>
</case>
