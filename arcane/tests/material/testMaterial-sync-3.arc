<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Material Synchronisation 3</titre>
  <description>Test Material Synchronisation 3</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage amr-type="3" nb-ghostlayer="3" ghostlayer-builder-version="3">
  <meshgenerator>
   <cartesian>
    <nsd>2 2</nsd>
    <origine>0.0 0.0</origine>
    <lx nx='16'>16.0</lx>
    <ly ny='16'>16.0</ly>
   </cartesian>
  </meshgenerator>
 </maillage>

 <module-test-unitaire>
  <test name="MeshMaterialSyncUnitTest">
   <nb-material>24</nb-material>
  </test>
 </module-test-unitaire>
</cas>
