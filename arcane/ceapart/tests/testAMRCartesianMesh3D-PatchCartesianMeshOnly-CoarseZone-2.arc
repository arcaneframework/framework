<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test CartesianMesh 3D PatchCartesianMeshOnly Coarse Zone (Variant 2)</title>

    <description>
      Test du raffinement d'un maillage cartesian 3D avec le type d'AMR PatchCartesianMeshOnly
      puis du dé-raffinement de certaines zones (avec renumérotation version 1)
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
    <renumber-patch-method>1</renumber-patch-method>

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

    <expected-number-of-cells-in-patchs>1000 3880</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>194910d2bfdfb101b23a3ce372f4e60d</nodes-uid-hash>
    <faces-uid-hash>145355f865846e248eea6a3458cb5cd9</faces-uid-hash>
    <cells-uid-hash>b025a6b3ac6545c730498d9967a65511</cells-uid-hash>
    <nodes-direction-hash>1f7be582f64304e490f0f49beeef8bda</nodes-direction-hash>
    <faces-direction-hash>f3073821e092607d071025c60b5881d7</faces-direction-hash>
    <cells-direction-hash>abb09d4f40fc4b799a19c46339118f2e</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter"/>
  </arcane-checkpoint>
</case>
