# AMR for Cartesian Meshes {#arcanedoc_entities_amr_cartesianmesh}

[TOC]

This subsection explains the AMR part of %Arcane for Cartesian Meshes.

## Introduction {#arcanedoc_entities_amr_cartesianmesh_intro}

Two types of AMR are available in %Arcane:

- Cell-based AMR for unstructured (and Cartesian) meshes (amr-type=1),
- Patch-based AMR for Cartesian meshes (amr-type=3).

AMR type 1 allows refining one or more meshes by providing an array of localIds.

```cpp
mesh()->modifier()->flagCellToRefine(cells_local_id);
mesh()->modifier()->adapt();
```
For Cartesian meshes, several methods are available to encapsulate the
non-structured AMR methods and allow for the creation of patches.

\note Nodes of meshes common between levels are not duplicated in AMR type 1.
They are in AMR type 3.

\image html amr_0.webp

For Cartesian meshes, AMR type 3 is also available and introduces a new API.
The user marks meshes to refine, and then %Arcane handles determining regular
patches as well as overlap meshes.
The operation is more guided.

To use AMR type 3, the face numbering must be changed. Currently, the change is
made during the first call to `Arcane::ICartesianMesh::computeDirections()`.
Furthermore, patch numbering can be managed by the class
`Arcane::CartesianMeshNumberingMng`.

The choice of AMR type to use is made in the dataset:

```xml

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
```

\note It is not yet possible to specify the AMR type with the new version of the
Cartesian mesh generator.

<br>

Table of Contents for this subsection:

1. \subpage arcanedoc_entities_amr_cartesianmesh_notions <br>
   Explanation of some concepts provided by AMR.</br>

2. \subpage arcanedoc_entities_amr_cartesianmesh_working <br>
   Explanation of how AMR works in %Arcane.</br>



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_entities_snippet_cartesianmesh
</span>
<span class="next_section_button">
\ref arcanedoc_entities_amr_cartesianmesh_notions
</span>
</div>
