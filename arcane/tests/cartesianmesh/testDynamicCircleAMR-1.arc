<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>DynamicCircleAMR (Variant 1)</title>
    <timeloop>DynamicCircleAMRLoop</timeloop>
  </arcane>

  <mesh amr-type="3">
    <meshgenerator>
      <cartesian>
        <nsd>2 2</nsd>
        <origine>0.0 0.0</origine>
        <lx nx='16'>64.0</lx>
        <ly ny='16'>64.0</ly>
      </cartesian>
    </meshgenerator>
  </mesh>

  <dynamic-circle-a-m-r>
    <nb-levels-max>3</nb-levels-max>
  </dynamic-circle-a-m-r>
</case>
