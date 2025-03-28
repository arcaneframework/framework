<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>DirectExec</titre>
  <description>DirectExec</description>
  <boucle-en-temps>ArcaneDirectExecutionLoop</boucle-en-temps>
 </arcane>
 <maillage utilise-unite="0">
  <fichier>sphere2.mli2</fichier>
 </maillage>
 <execution-directe>
  <tool name='ArcaneCasePartitioner'>
   <nb-partie-decoupees>24</nb-partie-decoupees>
   <correspondance>true</correspondance>
   <nb-couches-fantomes>1</nb-couches-fantomes>
   <bibliotheque>Metis</bibliotheque>
   <nom-service-ecriture>Lima</nom-service-ecriture>
  </tool>
 </execution-directe>
</cas>
