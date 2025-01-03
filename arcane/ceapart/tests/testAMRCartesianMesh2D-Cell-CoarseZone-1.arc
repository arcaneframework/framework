<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D Cell Coarse Zone (Variant 1)</titre>

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
        <lx nx='5' prx='1.0'>5.0</lx>
        <ly ny='5' pry='1.0'>5.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>1</renumber-patch-method>

    <refinement-2d>
      <position>1.0 1.0</position>
      <length>3.0 3.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>2.0 2.0</position>
      <length>1.0 1.0</length>
    </refinement-2d>

    <coarse-zone-2d>
      <position>2.0 2.0</position>
      <length>1.0 1.0</length>
    </coarse-zone-2d>
    <coarse-zone-2d>
      <position>2.0 2.0</position>
      <length>1.0 1.0</length>
    </coarse-zone-2d>

    <expected-number-of-cells-in-patchs>25 32</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>419ae016a6188b6fe2fce33111a0727b</nodes-uid-hash>
    <faces-uid-hash>f21c0a9a3f794391796fced1db892419</faces-uid-hash>
    <cells-uid-hash>77ff9dc6c92dc78592a5a0f522422acb</cells-uid-hash>
    <nodes-direction-hash>ef8d2321dfcefcd64230da9a344144b0</nodes-direction-hash>
    <faces-direction-hash>639237c778c0ccbb6d41e2da46f42e4d</faces-direction-hash>
    <cells-direction-hash>88908c6b1166b61acb92923029042140</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
