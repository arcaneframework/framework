<?xml version='1.0' ?><!-- -*- SGML -*- -->
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test Arcane 1</title>
    <description>Test Arcane 1</description>
    <timeloop>CaseFunctionTester</timeloop>
  </arcane>

  <mesh>
    <meshgenerator><sod><x>4</x><y>2</y><z>2</z></sod></meshgenerator>
  </mesh>

 <case-function-tester>
  <real-time-multiply-2 function="FuncTimeMultiply2">1.5</real-time-multiply-2>
  <int-iter-multiply-3 function="FuncIterMultiply3">1</int-iter-multiply-3>
  <real-norm-l2 function="FuncStandardRealReal3NormL2">1</real-norm-l2>
 </case-function-tester>

</case>
