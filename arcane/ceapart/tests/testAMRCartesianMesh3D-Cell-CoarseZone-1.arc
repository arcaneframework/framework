<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 3D Cell Coarse Zone (Variant 1)</titre>

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
        <lx nx='5' prx='1.0'>5.0</lx>
        <ly ny='5' pry='1.0'>5.0</ly>
        <lz nz='5' prz='1.0'>5.0</lz>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>1</renumber-patch-method>

    <refinement-3d>
      <position>1.0 1.0 1.0</position>
      <length>3.0 3.0 3.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>2.0 2.0 2.0</position>
      <length>1.0 1.0 1.0</length>
    </refinement-3d>

    <coarse-zone-3d>
      <position>2.0 2.0 2.0</position>
      <length>1.0 1.0 1.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>2.0 2.0 2.0</position>
      <length>1.0 1.0 1.0</length>
    </coarse-zone-3d>

    <expected-number-of-cells-in-patchs>125 208</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>e3774827d2b951cb6b7c36e3cdd04540</nodes-uid-hash>
    <faces-uid-hash>b0a24dbeeea1bba8b887155109d5a312</faces-uid-hash>
    <cells-uid-hash>0e51a40d9d1a79d115f4bb5ddeefd827</cells-uid-hash>
    <nodes-direction-hash>7e4d7e87ae23945f827ee8ade330760d</nodes-direction-hash>
    <faces-direction-hash>a4a2370619cd30637fec8a9baea03b9b</faces-direction-hash>
    <cells-direction-hash>9df145d5cd32ba170267fd6044210e03</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
