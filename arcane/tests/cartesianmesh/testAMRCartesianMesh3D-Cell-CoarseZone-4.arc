<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 3D Cell Coarse Zone (Variant 4)</titre>

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

    <expected-number-of-cells-in-patchs>1000 3880</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>aebfb1a98f6e90d452ac6f007628564a</nodes-uid-hash>
    <faces-uid-hash>dab2908cc09b1572c08793666b2ebe10</faces-uid-hash>
    <cells-uid-hash>5148e6e7af0ae779041a8bb42a7ca637</cells-uid-hash>
    <nodes-direction-hash>1e19f0aa5f439cfa4105c42d905bb1a6</nodes-direction-hash>
    <faces-direction-hash>fbdc6e058df75c4e0b7f5508111a9826</faces-direction-hash>
    <cells-direction-hash>1b83ea2ac7d60e6b4723162d6c09da68</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
