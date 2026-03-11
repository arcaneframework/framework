<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>AMRPatchTester (Variant 2 (3D))</title>
    <timeloop>AMRPatchTesterLoop</timeloop>
    <modules>
      <module name="ArcaneCheckpoint" active="true"/>
    </modules>
  </arcane>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasicCheckpointWriter"/>
    <do-dump-at-end>true</do-dump-at-end>
  </arcane-checkpoint>

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
