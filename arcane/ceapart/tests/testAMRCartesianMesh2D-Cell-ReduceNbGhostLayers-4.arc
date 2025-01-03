<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D Cell Reduce Number of Ghost Layers (Variant 4)</titre>

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
        <lx nx='12' prx='1.0'>12.0</lx>
        <ly ny='12' pry='1.0'>12.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>1</renumber-patch-method>
    <coarse-at-init>true</coarse-at-init>

    <reduce-nb-ghost-layers>
      <level>1</level>
      <nb-ghost-layers>2</nb-ghost-layers>
    </reduce-nb-ghost-layers>

    <refinement-2d>
      <position>5.0 2.0</position>
      <length>2.0 2.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>1.0 5.0</position>
      <length>2.0 2.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>0.0 10.0</position>
      <length>4.0 2.0</length>
    </refinement-2d>

    <coarse-zone-2d>
      <position>2.0 2.0</position>
      <length>2.0 2.0</length>
    </coarse-zone-2d>
    <coarse-zone-2d>
      <position>8.0 2.0</position>
      <length>2.0 2.0</length>
    </coarse-zone-2d>
    <coarse-zone-2d>
      <position>4.0 8.0</position>
      <length>6.0 2.0</length>
    </coarse-zone-2d>
    <coarse-zone-2d>
      <position>2.0 5.0</position>
      <length>1.0 1.0</length>
    </coarse-zone-2d>
    <coarse-zone-2d>
      <position>2.0 10.0</position>
      <length>2.0 1.0</length>
    </coarse-zone-2d>

    <expected-number-of-cells-in-patchs>36 124 16 12 24</expected-number-of-cells-in-patchs>
    <hash-with-ghost>true</hash-with-ghost>
    <nodes-uid-hash>64f70706d9e8f4bf6dfb874823b667f5</nodes-uid-hash>
    <faces-uid-hash>4391cd8a80530489023daaecce254c45</faces-uid-hash>
    <cells-uid-hash>aac8e02ea0b23cc12545caacbb80acce</cells-uid-hash>
    <nodes-direction-hash>a59dd6b7d1d9d732b91965073aa20a6a</nodes-direction-hash>
    <faces-direction-hash>2d646f6f6e5d93ca6ab3f6db0c87b2f1</faces-direction-hash>
    <cells-direction-hash>da8019a05f93f1e90580d32549254fb9</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
