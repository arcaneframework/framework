<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D Cell Coarse Zone (Variant 2)</titre>

    <description>
      Test du raffinement d'un maillage cartesian 2D avec le type d'AMR Cell puis du dé-raffinement de certaines zones
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
        <nsd>2 2</nsd>
        <origine>0.0 0.0</origine>
        <lx nx='10' prx='1.0'>10.0</lx>
        <ly ny='10' pry='1.0'>10.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

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
    <nodes-uid-hash>fc8dd8d6d47a03fd7abe97fde5e74b94</nodes-uid-hash>
    <faces-uid-hash>2a8a808d2c4ac5f760812c9b5b16c3f0</faces-uid-hash>
    <cells-uid-hash>9b275bfdff8e4485ab746bab360d02d3</cells-uid-hash>
    <nodes-direction-hash>66ab7970cec55512c36f8b50df7e6d0a</nodes-direction-hash>
    <faces-direction-hash>c017f7f8020fab9ae8a749a05540cffc</faces-direction-hash>
    <cells-direction-hash>a7698709767a059c1670beac54f1c8c3</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
