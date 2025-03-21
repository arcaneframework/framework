<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test CartesianMesh 2D PatchCartesianMeshOnly Coarse Zone (Variant 4)</title>

    <description>
      Test du raffinement d'un maillage cartesian 2D avec le type d'AMR PatchCartesianMeshOnly
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
        <nsd>2 2</nsd>
        <origine>0.0 0.0</origine>
        <lx nx='10' prx='1.0'>10.0</lx>
        <ly ny='10' pry='1.0'>10.0</ly>
      </cartesian>
    </meshgenerator>
  </mesh>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>

    <refinement-2d>
      <position>1.0 1.0</position>
      <length>8.0 8.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>3.0 3.0</position>
      <length>4.0 4.0</length>
    </refinement-2d>

    <coarse-zone-2d>
      <position>3.0 3.0</position>
      <length>2.0 2.0</length>
    </coarse-zone-2d>
    <coarse-zone-2d>
      <position>5.0 5.0</position>
      <length>2.0 2.0</length>
    </coarse-zone-2d>
    <coarse-zone-2d>
      <position>3.0 5.0</position>
      <length>2.0 2.0</length>
    </coarse-zone-2d>
    <coarse-zone-2d>
      <position>5.0 3.0</position>
      <length>2.0 2.0</length>
    </coarse-zone-2d>

    <coarse-zone-2d>
      <position>3.0 3.0</position>
      <length>3.0 3.0</length>
    </coarse-zone-2d>

    <expected-number-of-cells-in-patchs>100 220</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>bc4723a3ae6b84325bb17509f94d624b</nodes-uid-hash>
    <faces-uid-hash>1d4aba023756b0548b078f7915e2001e</faces-uid-hash>
    <cells-uid-hash>8874beeeeb07e91d1d49868bcce26b21</cells-uid-hash>
    <nodes-direction-hash>e2a8b311b668f6c7374b7339ccfeaf40</nodes-direction-hash>
    <faces-direction-hash>7e3271c244da2ebf50ab4abbcd870c67</faces-direction-hash>
    <cells-direction-hash>adf4c70f22ce75695fcf9e85eddcdc2c</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter"/>
  </arcane-checkpoint>
</case>
