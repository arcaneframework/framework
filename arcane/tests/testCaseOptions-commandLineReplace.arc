<?xml version='1.0' ?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test Arcane 1</title>
    <description>Test Arcane 1</description>
    <timeloop>CaseOptionsTester2</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Sod3D">
        <x>4</x><y>2</y><z>2</z>
      </generator>
    </mesh>
  </meshes>

  <case-options-tester>
    <test-id>@TestId@</test-id>
    <max-iteration>20</max-iteration>
    <simple-real-unit2>@SimpleRealUnit2@</simple-real-unit2>
    <simple-real3>@SimpleReal3@</simple-real3>

    <simple-string-multiple>toto1</simple-string-multiple>
    <simple-string-multiple>@SimpleStringMultiple@</simple-string-multiple>

    <simple-real-array-multi>3.0 4.1 5.6</simple-real-array-multi>

    <simple-with-standard-function>4.5</simple-with-standard-function>

    <simple-enum-function>enum1</simple-enum-function>

    <post-processor1 name="@PostProcessor1FrName@" mesh-name="Mesh0">
    </post-processor1>

    <post-processor1 name="UCDPostProcessor"/>

    <post-processor1>
      <use-collective-write>true</use-collective-write>
      <max-write-size>50</max-write-size>
    </post-processor1>

    <post-processor2 name="Ensight7PostProcessor">
      <fileset-size>12</fileset-size>
      <binary-file>false</binary-file>
    </post-processor2>
    <post-processor2 name="UCDPostProcessor"/>
    <post-processor2 name="Ensight7PostProcessor">
      <fileset-size>5</fileset-size>
      <binary-file>false</binary-file>
    </post-processor2>

    <post-processor3 name="Ensight7PostProcessor">
      <fileset-size>32</fileset-size>
      <binary-file>false</binary-file>
    </post-processor3>
    <post-processor3 name="Ensight7PostProcessor" mesh-name="Mesh0">
      <fileset-size>32</fileset-size>
      <binary-file>false</binary-file>
    </post-processor3>
    <post-processor4 name="Ensight7PostProcessor">
      <fileset-size>64</fileset-size>
      <binary-file>false</binary-file>
    </post-processor4>

    <complex1>
      <simple-real-2>3</simple-real-2>
      <simple-real-2-multi>5.2</simple-real-2-multi>
      <simple-real-2-multi>2.3</simple-real-2-multi>
      <simple-real3-2>3.0 2.0 4.0</simple-real3-2>
      <simple-integer-2>4</simple-integer-2>
      <simple-enum-2>enum2</simple-enum-2>
      <extended-real-int-2>enum1</extended-real-int-2>
      <complex1-sub>
        <sub-test1>2.0 3.0</sub-test1>
        <sub-test2>1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0</sub-test2>
      </complex1-sub>
    </complex1>
    <complex2>
      <simple-bool-c2>true</simple-bool-c2>
      <simple-real-c2>1.</simple-real-c2>
      <simple-integer-c2>4</simple-integer-c2>
      <simple-enum-c2>enum1</simple-enum-c2>
      <extended-real-int-c2>enum1</extended-real-int-c2>
    </complex2>
    <complex2>
      <simple-bool-c2>true</simple-bool-c2>
      <simple-real-c2>3</simple-real-c2>
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

</case>
