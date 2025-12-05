<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test CartesianMesh 3D PatchCartesianMeshOnly Coarse Zone (Variant 3)</title>

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
        <lx nx='5' prx='1.0'>5.0</lx>
        <ly ny='5' pry='1.0'>5.0</ly>
        <lz nz='5' prz='1.0'>5.0</lz>
      </cartesian>
    </meshgenerator>
  </mesh>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>

    <refinement-3d>
      <position>0.0 1.0 1.0</position>
      <length>3.0 3.0 3.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>0.0 2.0 2.0</position>
      <length>1.0 1.0 1.0</length>
    </refinement-3d>

    <coarse-zone-3d>
      <position>0.0 2.0 2.0</position>
      <length>1.0 1.0 1.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>0.0 2.0 2.0</position>
      <length>1.0 1.0 1.0</length>
    </coarse-zone-3d>

    <!--    <expected-number-of-cells-in-patchs>125 208</expected-number-of-cells-in-patchs>-->
    <expected-number-of-cells-in-patchs>125 16 16 24 24 32 48 48</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>549c906b4835458793357c764e285d6c</nodes-uid-hash>
    <faces-uid-hash>2470beb5b4e98e14b204a2150718c269</faces-uid-hash>
    <cells-uid-hash>67073338e696ff7e73e429ed9965fd28</cells-uid-hash>
    <nodes-direction-hash>cfe748e9a5ab3902c1fb7fee89f916b4</nodes-direction-hash>
    <faces-direction-hash>750c9dd4600a3b72e107098a257c0a63</faces-direction-hash>
    <cells-direction-hash>8edd0c2fa02d38971e90684f90b01e40</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter"/>
  </arcane-checkpoint>
</case>
