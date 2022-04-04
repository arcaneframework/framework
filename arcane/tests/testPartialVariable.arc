<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <modules>
   <module name="ArcaneLoadBalance" active="true" />
  </modules>
  <titre>Test Variables Partielles</titre>
  <description>Test des variables partielles</description>
  <boucle-en-temps>PartialVariableTestLoop</boucle-en-temps>
 </arcane>

 <maillage>
  <meshgenerator><sod><x>20</x><y>4</y><z>4</z></sod></meshgenerator>
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

 <!--arcane-protections-reprises>
  <en-fin-de-calcul>true</en-fin-de-calcul>
  <service-protection name="ArcaneHdf5MultiCheckpoint" />
 </arcane-protections-reprises-->

 <arcane-equilibrage-charge>
   <actif>true</actif>
   <partitionneur name="DefaultPartitioner"/>
   <periode>3</periode>
   <statistiques>true</statistiques>
   <desequilibre-maximal>0</desequilibre-maximal>
   <temps-cpu-minimal>0</temps-cpu-minimal>
 </arcane-equilibrage-charge>

</cas>
