<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Tube a choc de Sod</titre>
    <boucle-en-temps>AdiProjectionTestLoop</boucle-en-temps>

     <modules>
      <module name="ArcanePostProcessing" active="true" />
    </modules>
 <configuration><parametre name="NotParallel" value="true" /></configuration></arcane>


<maillage>
 <meshgenerator>
   <sod>
     <x delta='0.05'>500</x>
     <y delta='0.25'>200</y>
 </sod>
 </meshgenerator>
 <initialisation>
  <variable nom="Density" valeur="1." groupe="ZG" />
  <variable nom="Pressure" valeur="1." groupe="ZG" />
  <variable nom="Density" valeur="0.125" groupe="ZD" />
  <variable nom="Pressure" valeur="0.1" groupe="ZD" />
 </initialisation>
</maillage> 

 <arcane-post-traitement>
   <depouillement>
    <!-- <variable>VolumeFlux</variable> -->
    <variable>Pressure</variable>
    <variable>Density</variable>
    <variable>OldDensity</variable>
    <variable>Velocity</variable>
    <variable>InternalEnergy</variable>
    <variable>NodalMassFluxLeft</variable>
    <variable>NodalMassFluxRight</variable>
    <variable>MassFluxLeft</variable>
    <variable>MassFluxRight</variable>
    <variable>DeltaMass</variable>
    <variable>NodalDensity</variable>
    <variable>OldNodalDensity</variable>
    <variable>PressureGradient</variable>
    <groupe>AllFaces</groupe>
    <groupe>ZG</groupe>
    <groupe>ZD</groupe>
    </depouillement>
   <periode-sortie>0</periode-sortie>
 </arcane-post-traitement>

</cas>