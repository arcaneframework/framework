<?xml version='1.0' ?><!-- -*- SGML -*- -->
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test Arcane 1</titre>
    <description>Test Arcane 1</description>
    <boucle-en-temps>UnitTest</boucle-en-temps>
  </arcane>

  <maillage>
    <meshgenerator><sod><x>4</x><y>2</y><z>2</z></sod></meshgenerator>
    <initialisation/>
  </maillage>

  <fonctions>
    <table nom='test-real-linear' parametre='temps' valeur='reel' interpolation='lineaire'>
      <valeur> <x>0.0</x> <y>2.0</y> </valeur>
      <valeur> <x>4.0</x> <y>7.0</y> </valeur>
      <valeur> <x>5.0</x> <y>31.</y> </valeur>
      <valeur> <x>6.0</x> <y>50.0</y> </valeur>
      <valeur> <x>10.0</x><y>-1.0</y> </valeur>
      <valeur> <x>14.0</x><y>-3.0</y> </valeur>
    </table>
  </fonctions>

  <module-test-unitaire>
    <test name="CaseFunctionUnitTest" />
  </module-test-unitaire>

</cas>
