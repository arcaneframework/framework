<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Configuration</titre>
  <description>Test Configuration</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
  <configuration>
   <add name="TestConfig2" value="-25" />
  </configuration>
 </arcane>

 <maillage>
  <meshgenerator><sod><x>10</x><y>5</y><z>5</z></sod></meshgenerator>
 </maillage>

 <module-test-unitaire>
  <test name="ConfigurationUnitTest" />
 </module-test-unitaire>

</cas>
