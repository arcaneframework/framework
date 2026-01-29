<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D Cell Reduce Number of Ghost Layers (Variant 1)</titre>

    <description>
      Test du raffinement/dé-raffinement d'un maillage cartesian 2D avec le type d'AMR Cell
      puis d'une réduction du nombre de couche de mailles fantômes d'un niveau raffiné.
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

    <expected-number-of-cells-in-patchs>64 256</expected-number-of-cells-in-patchs>
    <hash-with-ghost>true</hash-with-ghost>
    <nodes-uid-hash>9c80eb5defb7c026d9b642a79caf371f</nodes-uid-hash>
    <faces-uid-hash>b85a7a301c7e9ab96b90b8ec52a25c6b</faces-uid-hash>
    <cells-uid-hash>0990d79ee6a99514ab29d2b60ab82737</cells-uid-hash>
    <nodes-direction-hash>0b12dd5e5509aafe9585cebac983b2e5</nodes-direction-hash>
    <faces-direction-hash>6356bcc2ca5416aef40ab149dfc8e306</faces-direction-hash>
    <cells-direction-hash>589c2d77edb6d088a308a8d2e99cb961</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
