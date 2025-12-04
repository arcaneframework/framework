<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test CartesianMesh 2D PatchCartesianMeshOnly Coarse Zone (Variant 2)</title>

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

    <!--    <expected-number-of-cells-in-patchs>100 220</expected-number-of-cells-in-patchs>-->
    <expected-number-of-cells-in-patchs>100 36 48 64 72</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>bc4723a3ae6b84325bb17509f94d624b</nodes-uid-hash>
    <!-- Hash avant renumérotation niveau 0. -->
    <!--    <faces-uid-hash>1d4aba023756b0548b078f7915e2001e</faces-uid-hash>-->
    <faces-uid-hash>0eb269ca13f2241a9922ed69e371795b</faces-uid-hash>
    <cells-uid-hash>8874beeeeb07e91d1d49868bcce26b21</cells-uid-hash>
    <nodes-direction-hash>37f20b21b2eb8c41d589128e9f7c2ab1</nodes-direction-hash>
    <!--    <faces-direction-hash>9976902a8decf9d761d200c3d4b135e7</faces-direction-hash>-->
    <faces-direction-hash>e63ff700fb353ae950cc1837ed766e7f</faces-direction-hash>
    <cells-direction-hash>5f9f718794bdc870ba496d14f4b7bd33</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter"/>
  </arcane-checkpoint>
</case>
