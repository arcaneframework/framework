<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 3D Cell Coarse Zone (Variant 2)</titre>

    <description>
      Test du raffinement d'un maillage cartesian 3D avec le type d'AMR Cell puis du dé-raffinement de certaines zones
    </description>

    <boucle-en-temps>AMRCartesianMeshTestLoop</boucle-en-temps>

    <modules>
      <module name="ArcanePostProcessing" active="true"/>
      <module name="ArcaneCheckpoint" active="true"/>
    </modules>

  </arcane>

  <arcane-post-traitement>
    <periode-sortie>1</periode-sortie>
    <depouillement>
      <variable>Density</variable>
      <variable>NodeDensity</variable>
      <groupe>AllCells</groupe>
      <groupe>AllNodes</groupe>
      <groupe>AllFacesDirection0</groupe>
      <groupe>AllFacesDirection1</groupe>
    </depouillement>
  </arcane-post-traitement>


  <maillage amr="true">
    <meshgenerator>
      <cartesian>
        <nsd>2 2 2</nsd>
        <origine>0.0 0.0 0.0</origine>
        <lx nx='10' prx='1.0'>10.0</lx>
        <ly ny='10' pry='1.0'>10.0</ly>
        <lz nz='10' prz='1.0'>10.0</lz>
      </cartesian>
    </meshgenerator>
  </maillage>

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
    <nodes-uid-hash>bc24725664617cfaf545293bbc684ec0</nodes-uid-hash>
    <faces-uid-hash>145355f865846e248eea6a3458cb5cd9</faces-uid-hash>
    <cells-uid-hash>b025a6b3ac6545c730498d9967a65511</cells-uid-hash>
    <nodes-direction-hash>476c5de42b0852497089a558c7bdcc6f</nodes-direction-hash>
    <faces-direction-hash>f3073821e092607d071025c60b5881d7</faces-direction-hash>
    <cells-direction-hash>abb09d4f40fc4b799a19c46339118f2e</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
