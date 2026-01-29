<?xml version='1.0' encoding='UTF-8'?>
<case codeversion="1.0" codename="ArcaneTest" xml:lang="en">
  <arcane>
    <title>Test unitaire de la class CartesianPatch</title>
    <timeloop>UnitTest</timeloop>
    <configuration>
      <parametre name="NotCheckpoint" value="true"/>
      <parametre name="NotParallel" value="true"/>
    </configuration>
  </arcane>
  <!-- ***************************************************************** -->
  <!-- test unitaire      -->
  <unit-test-module>
    <xml-test name="UnitTestCartesianMeshPatch">
    </xml-test>
  </unit-test-module>
  <!-- ***************************************************************** -->
  <!--Definition du maillage cartesien -->
  <mesh amr="true" nb-ghostlayer="3" ghostlayer-builder-version="3" use-unit="true">
    <meshgenerator>
      <cartesian>
        <nsd>1 1</nsd>
        <origine>0.0 0.0 0.0</origine>
        <lx nx="10" prx="1.0">1.0</lx>
        <ly ny="5" pry="1.0">1.0</ly>
      </cartesian>
    </meshgenerator>
  </mesh>
</case>
