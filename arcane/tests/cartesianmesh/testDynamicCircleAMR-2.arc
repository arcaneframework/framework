<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>DynamicCircleAMR (Variant 2 (Variant 1 3D))</title>
    <timeloop>DynamicCircleAMRLoop</timeloop>
  </arcane>

  <mesh amr-type="3">
    <meshgenerator>
      <cartesian>
        <nsd>2 2 2</nsd>
        <origine>0.0 0.0 0.0</origine>
        <lx nx='8'>64.0</lx>
        <ly ny='8'>64.0</ly>
        <lz nz='4'>64.0</lz>
      </cartesian>
    </meshgenerator>
  </mesh>

  <dynamic-circle-a-m-r>
    <nb-levels-max>3</nb-levels-max>
  </dynamic-circle-a-m-r>
</case>
