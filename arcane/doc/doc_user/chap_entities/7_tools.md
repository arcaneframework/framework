# External Mesh Management Tools {#arcanedoc_entities_tools}

[TOC]

%Arcane offers two tools for converting or partitioning a mesh.

## Mesh Partitioning Tool {#arcanedoc_entities_tools_arcane_partition_mesh}

The `arcane_partition_mesh` tool allows you to partition an existing mesh and
generate a mesh file per partition. It is a parallel tool that uses `ParMetis`
by default to perform the partitioning. This tool is installed in the `bin`
directory of the %Arcane installation.

Usage is as follows:

```
${ARCANE_INSTALL_ROOT}/bin/arcane_partition_mesh
  -n nb_proc
  -p nb_part
  --writer writer_service_name
  [--manifold-]
  [-Wp,arg1,arg2,...]
  mesh_file_name
```

Possible values for `writer_service_name` are services that implement
\arcane{IMeshWriter}. For example:

- `MshMeshWriter` (the default)
- `LimaMeshWriter`
- `VtkLegacyMeshWriter`: only for visualizing the result. The generated files
  cannot be used for code reading.

The input mesh file (`mesh_file_name`) can be in MSH format (extension `.msh`),
VTK history format (extension `.vtk`), or a format supported by `Lima`
(extension `.mli2`, `.mli`, `.unf`). When the input mesh is in MSH format, it is
not possible to know beforehand if the mesh is non-manifold. By default, %Arcane
assumes it is a manifold mesh. If this is not the case, you must specify the
`--manifold-` option in the command line.

The partitioner will run on `nb_proc` processes (via MPI) and will generate
`nb_part` partitions. The number of partitions must be a multiple of the number
of processes used. In the output, there will be one file per partition. The
output files will be in the form `CPU00000`, `CPU00001`, ... with the extension
corresponding to the mesh format (for example, `.msh` for the MSH format).

The `-Wp` option allows you to provide arguments for the parallel launcher
(`mpiexec` or `srun` in general). For example, the value `-Wp,-c,4` allows
adding `-c 4` to the parallel launcher.

### Mesh Format MSH Handling

\warning Reading and writing in MSH format has been experimental since version
3.16.0 of %Arcane and currently has a number of limitations.

If the input and output format is MSH, each file will contain the same
`$Entities` as the original format, as well as the periodicity information
(`$Periodics`) for the partition concerned.
The following limitations currently exist:

- Only order 1 entities are managed
- Periodicity information other than node pairs (master, slave) is consistent.
  Affine or entity information is not meaningful. For (master, slave) pairs, the
  pair will be present in the file if one of the two nodes is present in the
  partition.
- Information about bounding boxes for `$Entities` is not managed
- The original blocks of the MSH format are not necessarily saved, and the
  original `$Entities` numbers are not preserved.

### Using Pre-cut Files

TODO

### Usage Examples

The following example uses 2 processors to cut the `onesphere.msh` mesh into 4
partitions. The output file will be in VTK format.

```
./bin/arcane_partition_mesh -n 2 -p 4 --writer VtkLegacyMeshWriter onesphere.msh
ls
CPU00000.vtk
CPU00001.vtk
CPU00002.vtk
CPU00003.vtk
```

The following example uses 4 processors to cut the `mesh_with_loose_items.msh`
mesh into 12 partitions. The output file will be in MSH format. Since the input
mesh is non-manifold, the `--manifold-` option is used to specify this.

```
./bin/arcane_partition_mesh -n 4 -p 12 --writer MshMeshWriter --manifold- mesh_with_loose_items.msh
ls
CPU00000.msh
CPU00001.msh
...
CPU00010.msh
CPU00011.msh
CPU00012.msh
```

## Mesh Conversion Tools {#arcanedoc_entities_tools_arcane_convert_mesh}

The `arcane_convert_mesh` tool allows you to convert a mesh file from one format
to another.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_entities_geometric
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_execution_direct_execution
</span> -->
</div>
