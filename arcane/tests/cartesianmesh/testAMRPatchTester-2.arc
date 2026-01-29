<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>AMRPatchTester (Variant 2 (3D))</title>
    <timeloop>AMRPatchTesterLoop</timeloop>
  </arcane>

  <arcane-post-processing>
  </arcane-post-processing>

  <mesh amr-type="3">
    <meshgenerator>
      <cartesian>
        <nsd>2 2 2</nsd>
        <origine>0.0 0.0 0.0</origine>
        <lx nx='8'>64.0</lx>
        <ly ny='8'>64.0</ly>
        <lz nz='8'>64.0</lz>
      </cartesian>
    </meshgenerator>
  </mesh>

</case>
