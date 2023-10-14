<?xml version='1.0' ?><!-- -*- SGML -*- -->
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test Arcane 5</titre>
    <description>Test Arcane 5</description>
    <boucle-en-temps>CaseOptionsTester</boucle-en-temps>
  </arcane>

  <maillage>
    <fichier>sod.vtk</fichier>
  </maillage>
  <maillage>
    <fichier>plancher.msh</fichier>
  </maillage>

  <fonctions>
    <table nom='test-simple-enum' parametre='temps' valeur='string' interpolation='constant-par-morceaux'>
      <valeur><x>2.</x><y>enum1</y></valeur>
      <valeur><x>5.</x><y>enum2</y></valeur>
      <valeur><x>10.</x><y>enum1</y></valeur>
      <valeur><x>15.</x><y>enum2</y></valeur>
      <valeur><x>18.</x><y>enum1</y></valeur>
    </table>
    <table nom='test-time-real' parametre='temps' valeur='reel' comul='1.e2' interpolation='lineaire'>
      <valeur> <x>0.0</x> <y>2.0</y> </valeur>
      <valeur> <x>4.0</x> <y>7.0</y> </valeur>
      <valeur> <x>5.0</x> <y>31.</y> </valeur>
      <valeur> <x>6.0</x> <y>50.0</y> </valeur>
      <valeur> <x>10.0</x><y>-1.0</y> </valeur>
      <valeur> <x>14.0</x><y>-3.0</y> </valeur>
    </table>
    <table nom='test-time-real-2' parametre='temps' valeur='reel' comul='1.e2' deltat-coef='0.0' interpolation='lineaire'>
      <valeur> <x>0.0</x> <y>3.0</y> </valeur>
      <valeur> <x>4.0</x> <y>9.0</y> </valeur>
      <valeur> <x>5.0</x> <y>7.</y> </valeur>
      <valeur> <x>6.0</x> <y>2.0</y> </valeur>
      <valeur> <x>10.0</x><y>-1.0</y> </valeur>
      <valeur> <x>14.0</x><y>-3.0</y> </valeur>
    </table>
    <table nom='test-time-bool' parametre='temps' valeur='bool' interpolation='constant-par-morceaux'>
      <valeur><x>2.0</x><y>true</y></valeur>
      <valeur><x>5.0</x><y>false</y></valeur>
      <valeur><x>10.0</x><y>1</y></valeur>
      <valeur><x>15.0</x><y>0</y></valeur>
      <valeur><x>18.0</x><y>true</y></valeur>
    </table>
    <table nom='test-time-real3' parametre='temps' valeur='reel3' deltat-coef='0.0' interpolation='lineaire'>
      <valeur> <x>0.0</x> <y>3.0 1.0 5.0</y> </valeur>
      <valeur> <x>4.0</x> <y>9.0 2.0 7.0</y> </valeur>
      <valeur> <x>5.0</x> <y>7.1 1.5 3.4</y> </valeur>
      <valeur> <x>6.0</x> <y>2.0 7.2 4.9</y> </valeur>
      <valeur> <x>10.0</x><y>-1.0 -2.0 5.0</y> </valeur>
      <valeur> <x>14.0</x><y>-3.0  8.0 1.3</y> </valeur>
    </table>
  </fonctions>

  <case-options-tester>
    <test-id>6</test-id>
    <max-iteration>20</max-iteration>

    <simple-real>3.0</simple-real>
    <simple-real>3.0</simple-real>
    <simple-real-unit>4.2</simple-real-unit>
    <simple-real-unit2 fonction="test-time-real">0.0</simple-real-unit2>
    <simple-realarray-unit >2.2 2.3 0.1 3.5</simple-realarray-unit>
    <simple-real2>3.5 7.2</simple-real2>
    <simple-real2x2>3.1 3.0 2.9 2.7</simple-real2x2>
    <simple-real3x3>3.3 3.2 2.1 1.1 0.3 0.5 7.2 7.1 4.0</simple-real3x3>
    <simple-integer>4</simple-integer>
    <simple-int32>-23</simple-int32>
    <simple-int64>454653457457455474</simple-int64>
    <simple-bool>true</simple-bool>
    <simple-string>toto</simple-string>

    <simple-real-array>3.0 4.1 5.6</simple-real-array>

    <simple-real-array-multi>3.0 4.1 5.6</simple-real-array-multi>
    <simple-real-array-multi>4.0 1.1 7.3</simple-real-array-multi>

    <simple-integer-array>4 5 6 7</simple-integer-array>
    <simple-int64-array>454653457457455474 -453463634634634634</simple-int64-array>
    <simple-bool-array>true false false true</simple-bool-array>
    <simple-string-array>toto titi tata tutu tete</simple-string-array>

    <simple-with-standard-function fonction="std_func">4.5</simple-with-standard-function>

    <simple-enum>enum1</simple-enum>
    <simple-enum-function fonction="test-simple-enum">enum1</simple-enum-function>
    <!-- <simple-enum-function fonction="test-simple-enum">enum1</simple-enum-function> -->
    <post-processor1-fr name="Ensight7PostProcessor">
      <nb-temps-par-fichier>1</nb-temps-par-fichier>
      <fichier-binaire>false</fichier-binaire>
    </post-processor1-fr>
    <post-processor1-fr name="Ensight7PostProcessor"/>
    <post-processor1-fr name="Ensight7PostProcessor">
      <nb-temps-par-fichier>9</nb-temps-par-fichier>
      <fichier-binaire>false</fichier-binaire>
    </post-processor1-fr>

    <post-processor2 name="Ensight7PostProcessor">
      <nb-temps-par-fichier>12</nb-temps-par-fichier>
      <fichier-binaire>false</fichier-binaire>
    </post-processor2>
    <post-processor2 name="Ensight7PostProcessor"/>
    <post-processor2 name="Ensight7PostProcessor">
      <nb-temps-par-fichier>5</nb-temps-par-fichier>
      <fichier-binaire>false</fichier-binaire>
    </post-processor2>

    <post-processor3-fr name="Ensight7PostProcessor">
      <nb-temps-par-fichier>32</nb-temps-par-fichier>
      <fichier-binaire>false</fichier-binaire>
    </post-processor3-fr>
    <post-processor4 name="Ensight7PostProcessor">
      <nb-temps-par-fichier>64</nb-temps-par-fichier>
      <fichier-binaire>false</fichier-binaire>
    </post-processor4>
    <service-instance-test1 name="ServiceInterface1ImplTest" >
      <complex1>
        <simple-real-2>3.5</simple-real-2>
        <cell-group>Planchere</cell-group>
      </complex1>
      <post-processor1 name="Ensight7PostProcessor">
        <nb-temps-par-fichier>3</nb-temps-par-fichier>
        <fichier-binaire>true</fichier-binaire>
      </post-processor1>
      <multi-post-processor name="Ensight7PostProcessor">
        <nb-temps-par-fichier>7</nb-temps-par-fichier>
        <fichier-binaire>false</fichier-binaire>
      </multi-post-processor>
      <multi-post-processor name="Ensight7PostProcessor">
        <nb-temps-par-fichier>1</nb-temps-par-fichier>
        <fichier-binaire>false</fichier-binaire>
      </multi-post-processor>
    </service-instance-test1>
    <service-instance-test2>
      <complex5>
        <simple-real-2>4.6</simple-real-2>
        <cell-group>Planchere</cell-group>
      </complex5>
      <sub-service-instance name="ServiceInterface1ImplTest">
        <complex1>
          <simple-real-2>1.2</simple-real-2>
          <cell-group>Planchere</cell-group>
        </complex1>
        <post-processor1 name="Ensight7PostProcessor" />
      </sub-service-instance>
    </service-instance-test2>
    <complex1>
      <simple-real-2>3</simple-real-2>
      <simple-real-2-multi>5.2</simple-real-2-multi>
      <simple-real-2-multi>2.3</simple-real-2-multi>
      <simple-real3-2 fonction="test-time-real3">3.0 2.0 4.0</simple-real3-2>
      <simple-integer-2>4</simple-integer-2>
      <simple-enum-2>enum2</simple-enum-2>
      <extended-real-int-2>enum1</extended-real-int-2>
      <complex1-sub>
        <sub-test1>2.0 3.0</sub-test1>
        <sub-test2>1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0</sub-test2>
      </complex1-sub>
    </complex1>
    <complex2>
      <simple-bool-c2 fonction="test-time-bool">true</simple-bool-c2>
      <simple-real-c2 fonction="test-time-real">1.</simple-real-c2>
      <simple-integer-c2>4</simple-integer-c2>
      <simple-enum-c2>enum1</simple-enum-c2>
      <extended-real-int-c2>enum1</extended-real-int-c2>
    </complex2>
    <complex2>
      <simple-bool-c2>true</simple-bool-c2>
      <simple-real-c2 fonction="test-time-real-2">3</simple-real-c2>
      <simple-integer-c2>4</simple-integer-c2>
      <simple-enum-c2>enum1</simple-enum-c2>
      <extended-real-int-c2>enum1</extended-real-int-c2>
      <complex3>
        <simple-real-c3>3</simple-real-c3>
        <simple-integer-c3>4</simple-integer-c3>
        <simple-enum-c3>enum1</simple-enum-c3>
        <extended-real-int-c3>enum1</extended-real-int-c3>
        <timeloop-tester name="CheckpointTesterService">
          <nb-iteration>5</nb-iteration>
          <backward-period>3</backward-period>
        </timeloop-tester>
      </complex3>
      <complex3>
        <simple-real-c3>5</simple-real-c3>
        <simple-integer-c3>7</simple-integer-c3>
        <simple-enum-c3>enum2</simple-enum-c3>
        <extended-real-int-c3>enum2</extended-real-int-c3>
        <timeloop-tester name="CheckpointTesterService">
          <nb-iteration>12</nb-iteration>
          <backward-period>4</backward-period>
        </timeloop-tester>
      </complex3>
    </complex2>
    <extended-real-int>enum1</extended-real-int>
    <complex4>
      <simple-real>5.2</simple-real>
    </complex4>
    <complex4>
      <simple-real>5.2</simple-real>
    </complex4>
    <complex5>
      <simple-real>4.9</simple-real>xs
    </complex5>
    <complex5>
      <simple-real>4.9</simple-real>xs
    </complex5>
  </case-options-tester>
</cas>
