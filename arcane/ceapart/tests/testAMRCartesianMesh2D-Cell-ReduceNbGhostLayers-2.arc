<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D Cell Reduce Number of Ghost Layers (Variant 2)</titre>

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
        <lx nx='8' prx='1.0'>8.0</lx>
        <ly ny='8' pry='1.0'>8.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>1</renumber-patch-method>
    <reduce-nb-ghost-layers>
      <level>1</level>
      <nb-ghost-layers>4</nb-ghost-layers>
    </reduce-nb-ghost-layers>

    <refinement-2d>
      <position>1.0 1.0</position>
      <length>2.0 4.0</length>
    </refinement-2d>

    <expected-number-of-cells-in-patchs>64 32</expected-number-of-cells-in-patchs>
    <hash-with-ghost>true</hash-with-ghost>
    <nodes-uid-hash>5ab97ac37a25e08ce148397befdac15e</nodes-uid-hash>
    <faces-uid-hash>efc046f4cdd163112b2b6244b1d70e4f</faces-uid-hash>
    <cells-uid-hash>666cbb9c97ab2b2c5564ca6e4421e9e3</cells-uid-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
