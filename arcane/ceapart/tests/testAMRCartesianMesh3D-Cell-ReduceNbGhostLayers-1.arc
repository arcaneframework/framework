<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 3D Cell Reduce Number of Ghost Layers (Variant 1)</titre>

    <description>
      Test du raffinement/dé-raffinement d'un maillage cartesian 3D avec le type d'AMR Cell
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
        <nsd>2 2 2</nsd>
        <origine>0.0 0.0 0.0</origine>
        <lx nx='16' prx='1.0'>16.0</lx>
        <ly ny='16' pry='1.0'>16.0</ly>
        <lz nz='16' prz='1.0'>16.0</lz>
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

    <expected-number-of-cells-in-patchs>512 4096</expected-number-of-cells-in-patchs>
    <hash-with-ghost>true</hash-with-ghost>
    <nodes-uid-hash>58286b8761b5b06952e9b91cd624ac6f</nodes-uid-hash>
    <faces-uid-hash>a7a20812fb9de2a5f13faf40156975c7</faces-uid-hash>
    <cells-uid-hash>c5fbba8ee08d9ca9ee4968063a865027</cells-uid-hash>
    <nodes-direction-hash>eb3856382185cc9cca85f67dbacfad72</nodes-direction-hash>
    <faces-direction-hash>156e9e2e15a9f29ed22b70dda3a6afaf</faces-direction-hash>
    <cells-direction-hash>8befd1036bccf187ef9fb1d42b939f69</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
