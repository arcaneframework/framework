<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test AcceleratorReduce 1</titre>
  <description>Test AcceleratorReduce 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <meshgenerator><sod><x>100</x><y>5</y><z>5</z></sod></meshgenerator>
 </maillage>

 <module-test-unitaire>
   <test name="AcceleratorReduceUnitTest">
     <use-atomic>true</use-atomic>
   </test>
 </module-test-unitaire>

</cas>
