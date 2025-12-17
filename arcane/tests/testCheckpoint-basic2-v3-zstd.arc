<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Protections/Reprises</titre>
  <description>Test des protections/reprise avec le interne Arcane (Version 1)</description>
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
   <service-protection name="ArcaneBasic2CheckpointWriter">
     <format-version>3</format-version>
     <data-compressor name="zstdDataCompressor" />
   </service-protection>
   <periode>3</periode>
   <en-fin-de-calcul>false</en-fin-de-calcul>
 </arcane-protections-reprises>
</cas>
