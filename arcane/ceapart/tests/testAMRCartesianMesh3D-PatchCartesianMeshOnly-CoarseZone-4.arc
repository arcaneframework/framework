<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test CartesianMesh 3D PatchCartesianMeshOnly Coarse Zone (Variant 4)</title>

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
      <position>0.0 1.0 1.0</position>
      <length>8.0 8.0 8.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>0.0 3.0 3.0</position>
      <length>4.0 4.0 4.0</length>
    </refinement-3d>

    <coarse-zone-3d>
      <position>0.0 3.0 3.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>0.0 3.0 5.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>0.0 5.0 3.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>0.0 5.0 5.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>2.0 3.0 3.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>2.0 3.0 5.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>2.0 5.0 3.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>2.0 5.0 5.0</position>
      <length>2.0 2.0 2.0</length>
    </coarse-zone-3d>

    <coarse-zone-3d>
      <position>0.0 3.0 3.0</position>
      <length>3.0 3.0 3.0</length>
    </coarse-zone-3d>

    <!--    <expected-number-of-cells-in-patchs>1000 3880</expected-number-of-cells-in-patchs>-->
    <expected-number-of-cells-in-patchs>1000 360 432 576 720 768 1024</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>38ce8a131e14a4ef3db4e9ac0b1b40e6</nodes-uid-hash>
    <faces-uid-hash>dab2908cc09b1572c08793666b2ebe10</faces-uid-hash>
    <cells-uid-hash>5148e6e7af0ae779041a8bb42a7ca637</cells-uid-hash>
    <nodes-direction-hash>7853a0e71c60141f5ce3ef1611ed0ce8</nodes-direction-hash>
    <faces-direction-hash>19b26d1b513014c1c568ba812dc0c23f</faces-direction-hash>
    <cells-direction-hash>5f291689ed9db06199bfec171a2dee95</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter"/>
  </arcane-checkpoint>
</case>
