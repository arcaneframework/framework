<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Protections/Reprises</titre>
  <description>Test des protections/reprise avec le service HDF5 parallel (MPI)</description>
  <boucle-en-temps>BasicLoop</boucle-en-temps>
  <modules>
   <module name="ArcaneCheckpoint" actif="true" />
  </modules>
 </arcane>

 <maillage>
  <meshgenerator><sod><x>20</x><y>2</y><z>2</z></sod></meshgenerator>
  <initialisation />
 </maillage>

 <module-maitre>
  <service-global name="CheckpointTesterService">
   <nb-iteration>5</nb-iteration>
  </service-global>
 </module-maitre>

 <arcane-protections-reprises>
   <service-protection name="ArcaneHdf5MpiCheckpoint2" />
   <!-- <service-protection name="ArcaneHdf5Checkpoint2">
    <nombre-fichier-par-groupe>4</nombre-fichier-par-groupe>
  </service-protection> -->

  <periode>3</periode>
  <en-fin-de-calcul>false</en-fin-de-calcul>
 </arcane-protections-reprises>
</cas>
