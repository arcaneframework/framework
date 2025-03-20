<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test CartesianMesh 2D PatchCartesianMeshOnly Coarse Zone (Variant 2)</title>

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
        <lx nx='10' prx='1.0'>10.0</lx>
        <ly ny='10' pry='1.0'>10.0</ly>
      </cartesian>
    </meshgenerator>
  </mesh>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>1</renumber-patch-method>

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
    <nodes-uid-hash>adbff87895b55d9ce55b60fcab95c542</nodes-uid-hash>
    <faces-uid-hash>2a8a808d2c4ac5f760812c9b5b16c3f0</faces-uid-hash>
    <cells-uid-hash>9b275bfdff8e4485ab746bab360d02d3</cells-uid-hash>
    <nodes-direction-hash>3169601d3311a9bbd3f02dfe47ce263c</nodes-direction-hash>
    <faces-direction-hash>c017f7f8020fab9ae8a749a05540cffc</faces-direction-hash>
    <cells-direction-hash>a7698709767a059c1670beac54f1c8c3</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter"/>
  </arcane-checkpoint>
</case>
