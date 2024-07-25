<?xml version="1.0" ?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test Protections/Reprises</titre>
    <description>Test des protections/reprise avec module de verification</description>
    <boucle-en-temps>BasicLoop</boucle-en-temps>
    <modules>
      <module name="ArcaneCheckpoint" actif="true"/>
    </modules>
  </arcane>

  <maillage nb-ghostlayer="3" ghostlayer-builder-version="3">
    <fichier internal-partition="true">sod.vtk</fichier>
  </maillage>

  <module-maitre>
    <service-global name="CheckpointTesterService">
      <nb-iteration>5</nb-iteration>
    </service-global>
  </module-maitre>

  <arcane-protections-reprises>
    <en-fin-de-calcul>false</en-fin-de-calcul>
  </arcane-protections-reprises>

  <verificateur>
    <verification-active>true</verification-active>
    <generation>false</generation>
    <comparaison-parallele-sequentiel>true</comparaison-parallele-sequentiel>
    <verifier-service-name>ArcaneBasicVerifier3</verifier-service-name>
    <fichier-reference>checkpoint-test-verifier-file</fichier-reference>
    <fichiers-dans-dossier-output>true</fichiers-dans-dossier-output>
  </verificateur>
</cas>
