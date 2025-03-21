<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test CartesianMesh 2D PatchCartesianMeshOnly Coarse Zone (Variant 1)</title>

    <description>
      Test du raffinement d'un maillage cartesian 2D avec le type d'AMR PatchCartesianMeshOnly
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
        <nsd>2 2</nsd>
        <origine>0.0 0.0</origine>
        <lx nx='5' prx='1.0'>5.0</lx>
        <ly ny='5' pry='1.0'>5.0</ly>
      </cartesian>
    </meshgenerator>
  </mesh>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>1</renumber-patch-method>

    <refinement-2d>
      <position>1.0 1.0</position>
      <length>3.0 3.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>2.0 2.0</position>
      <length>1.0 1.0</length>
    </refinement-2d>

    <coarse-zone-2d>
      <position>2.0 2.0</position>
      <length>1.0 1.0</length>
    </coarse-zone-2d>
    <coarse-zone-2d>
      <position>2.0 2.0</position>
      <length>1.0 1.0</length>
    </coarse-zone-2d>

    <expected-number-of-cells-in-patchs>25 32</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>b85cb78ab44742ca5ae8d1440a38739d</nodes-uid-hash>
    <faces-uid-hash>f21c0a9a3f794391796fced1db892419</faces-uid-hash>
    <cells-uid-hash>77ff9dc6c92dc78592a5a0f522422acb</cells-uid-hash>
    <nodes-direction-hash>278dd927e27f254e9dd6e03788fe10ca</nodes-direction-hash>
    <faces-direction-hash>639237c778c0ccbb6d41e2da46f42e4d</faces-direction-hash>
    <cells-direction-hash>88908c6b1166b61acb92923029042140</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter"/>
  </arcane-checkpoint>
</case>
