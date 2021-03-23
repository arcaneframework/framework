<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="ArcaneTest" xml:lang="en">
  <arcane>
    <title>Test du coraffinement parallèle</title>
    <timeloop>UnitTest</timeloop>
  </arcane>
  <mesh>
    <file internal-partition='true'>plan.vtk</file>
  </mesh>

  <arcane-post-processing>
    <output-period>1</output-period>
    <output>
      <variable>ContactVoidRatio</variable>
      <variable>ContactRatio</variable>
      <variable>ContactCount</variable>
    </output>
  </arcane-post-processing>

  <unit-test-module>
    <test name="ParallelCorefinementTest">
      <master>ZMIN</master>
      <slave>ZMAX</slave>
      <box-tolerance>1</box-tolerance>
      
      <corefinement name="ParallelCorefinement">
        <surface-utils name="GeometryKernelSurfaceTools">
        </surface-utils>
      </corefinement>
      <xy-master-shift>0.10</xy-master-shift>
    </test>
  </unit-test-module>
</case>
