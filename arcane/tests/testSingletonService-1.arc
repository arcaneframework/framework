<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test SingletonService</titre>
  <description>Test SingletonService</description>
  <boucle-en-temps>SingletonServiceTestModuleLoop</boucle-en-temps>
  <configuration>
   <add name="TestConfig2" value="-25" />
   <add name="TestGlobalConfig3" value="9.3" />
  </configuration>
  <modules>
  </modules>
  <services>
    <service name="TestSingleton5" />
    <service name="TestSingleton6" active="true" />
    <service name="TestSingleton7" active="false" />
  </services>
 </arcane>

 <maillage>
   <meshgenerator><sod><x>2</x><y>2</y><z>10</z></sod></meshgenerator> 
  <initialisation />
 </maillage>

</cas>
