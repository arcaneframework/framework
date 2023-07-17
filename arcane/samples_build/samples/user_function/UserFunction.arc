<?xml version="1.0"?>
<case codename="UserFunction" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Sample</title>
    <timeloop>UserFunctionLoop</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <!-- You can use a file instead of a generator using the element
           <filename>/path/to/the/file</filename> instead of <generator/> --> 
      <generator name="Cartesian2D" >
        <nb-part-x>1</nb-part-x> 
        <nb-part-y>1</nb-part-y>
        <origin>0.0 0.0</origin>
        <x><n>20</n><length>2.0</length></x>
        <y><n>20</n><length>2.0</length></y>
      </generator>
    </mesh>
  </meshes>

  <functions>
    <external-assembly>
      <assembly-name>ExternalFunctions.dll</assembly-name>
      <class-name>UserFunctionSample.CaseFunctions</class-name>
    </external-assembly>
  </functions>

  <arcane-post-processing>
    <output-period>1</output-period>
    <output>
      <variable>NodeVelocity</variable>
    </output>
  </arcane-post-processing>

  <user-function>
    <node-velocity function="NodeVelocityFunc">0.0 0.0 0.0</node-velocity>
  </user-function>

</case>
