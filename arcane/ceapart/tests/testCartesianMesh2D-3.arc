<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh</titre>
    <description>Test des maillages cartesiens 2D avec une seule couche de maille en Y</description>

    <boucle-en-temps>CartesianMeshTestLoop</boucle-en-temps>

    <modules>
      <module name="ArcanePostProcessing" active="true"/>
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

  <maillage>
    <meshgenerator>
      <cartesian>
        <nsd>2 1</nsd>
        <origine>0.0 0.0</origine>
        <lx nx='4' prx='1.0'>4.0</lx>
        <ly ny='1' pry='1.0'>1.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

</cas>
