<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D Cell Reduce Number of Ghost Layers (Variant 3)</titre>

    <description>
      Test du raffinement/dé-raffinement d'un maillage cartesian 2D avec le type d'AMR Cell
      puis d'une réduction du nombre de couches de mailles fantômes d'un niveau raffiné.
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


  <maillage amr="true" nb-ghostlayer="3" ghostlayer-builder-version="3">
    <meshgenerator>
      <cartesian>
        <nsd>2 2</nsd>
        <origine>0.0 0.0</origine>
        <lx nx='16' prx='1.0'>16.0</lx>
        <ly ny='16' pry='1.0'>16.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>1</renumber-patch-method>
    <coarse-at-init>true</coarse-at-init>

    <reduce-nb-ghost-layers>
      <level>1</level>
      <nb-ghost-layers>4</nb-ghost-layers>
    </reduce-nb-ghost-layers>

    <coarse-zone-2d>
      <position>0.0 0.0</position>
      <length>2.0 16.0</length>
    </coarse-zone-2d>
    <coarse-zone-2d>
      <position>2.0 10.0</position>
      <length>14.0 6.0</length>
    </coarse-zone-2d>
    <coarse-zone-2d>
      <position>2.0 0.0</position>
      <length>14.0 2.0</length>
    </coarse-zone-2d>
    <coarse-zone-2d>
      <position>6.0 2.0</position>
      <length>10.0 8.0</length>
    </coarse-zone-2d>

    <expected-number-of-cells-in-patchs>64 32</expected-number-of-cells-in-patchs>
    <hash-with-ghost>true</hash-with-ghost>
    <nodes-uid-hash>77deb6106ef5e9af01995c0f23b8107b</nodes-uid-hash>
    <faces-uid-hash>84f3fadd4117122c7c0d1304ca88fcb5</faces-uid-hash>
    <cells-uid-hash>4771e20622bb9bc5ba16374beb405442</cells-uid-hash>
    <nodes-direction-hash>e7cc76f34db1eab9da795ef1872bd77f</nodes-direction-hash>
    <faces-direction-hash>0aec913c764d0bc9cf90c6a5af933691</faces-direction-hash>
    <cells-direction-hash>8439e3e858ebe40282a631083ed87878</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
