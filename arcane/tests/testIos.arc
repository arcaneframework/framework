<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test IOS Reader/Writer SOD->VTU->XMF->MSH->ARC</titre>
  <description>Lecture/Ecriture d'un cycle (SOD->VTU->XMF->MSH->ARC)</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage><meshgenerator><sod><x>100</x><y>5</y><z>5</z></sod></meshgenerator></maillage>
 <maillage><meshgenerator><simple><mode>1</mode></simple></meshgenerator></maillage>
 <maillage><meshgenerator><simple><mode>2</mode></simple></meshgenerator></maillage>

 
 <module-test-unitaire>
	<test name="IosUnitTest">

			<ecriture-vtu>true</ecriture-vtu>
			<ecriture-xmf>false</ecriture-xmf>
			<ecriture-msh>false</ecriture-msh>
		</test>
	</module-test-unitaire>
</cas>