<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Variables Partielles</titre>
  <description>Test des variables partielles</description>
  <boucle-en-temps>PartialVariableTestLoop</boucle-en-temps>
 </arcane>

 <maillage>
  <meshgenerator><sod><x>100</x><y>5</y><z>5</z></sod></meshgenerator>
  <initialisation />
 </maillage>

 <partial-variable-tester>
   <max-iteration>15</max-iteration>
 </partial-variable-tester>

 <arcane-post-traitement>
   <periode-sortie>1</periode-sortie>
   <!--format>Ensight7Gold</format-->
   <depouillement>
     <variable>Rank</variable>
     <variable>InitialRank</variable>
     <variable>PostCurrentRank</variable>
     <variable>PostInitialRank</variable>
     <variable>CellTemperature</variable>
     <variable>NodeTemperature</variable>
   </depouillement>
 </arcane-post-traitement>

</cas>
