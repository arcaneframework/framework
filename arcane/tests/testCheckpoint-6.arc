<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Protections/Reprises</titre>
  <description>Test des protections/reprise changement des informations sur les mailles fantomes</description>
  <boucle-en-temps>BasicLoop</boucle-en-temps>
  <modules>
   <module name="ArcaneCheckpoint" actif="true" />
  </modules>
 </arcane>

 <!-- <maillage nb-ghostlayer="5" ghostlayer-builder-version="3"> -->
 <maillage nb-ghostlayer="5" ghostlayer-builder-version="3">
  <fichier internal-partition="true" >sod.vtk</fichier>
 </maillage>

<module-maitre>
  <service-global name="CheckpointTesterService">
   <nb-iteration>5</nb-iteration>
  </service-global>
 </module-maitre>

 <arcane-protections-reprises>
  <!-- <service-protection name="ArcaneBasicCheckpointWriter" /> -->
  <service-protection name="ArcaneHdf5Checkpoint2" />
  <periode>3</periode>
  <en-fin-de-calcul>false</en-fin-de-calcul>
 </arcane-protections-reprises>
</cas>
