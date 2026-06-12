# Usage for materials {#arcanedoc_acceleratorapi_materials}

[TOC]

It is possible to use the %Arcane accelerator API to manage constituents.
Instances of the material manager \arcanemat{IMeshMaterialMng} use the execution
manager (\arcaneacc{Runner}) associated with the subdomain from which the mesh
originates. Constituent management will therefore be handled automatically using
the same execution environment as operations on classic mesh entities.

Using the accelerator API for materials is similar to using it for mesh
entities. The macro RUNCOMMAND_MAT_ENUMERATE() allows iteration over an
environment \arcanemat{IMeshEnvironment} or a material
\arcanemat{IMeshMaterial}.

The possible values for this macro are:

<table>
<tr>
<th>Iteration Type</th><th>Iterator Value</th>
<th>Container Type</th><th>Description</th>
</tr>

<tr>
<td>EnvAndGlobalCell</td>
<td>\arcanemat{EnvAndGlobalCellIteratorValue}</td>
<td>\arcanemat{IMeshEnvironment} <br></br>
\arcanemat{EnvCellVectorView}</td>
<td>Iteration over an environment allowing the retrieval of the local cell
number of the environment, the iteration index, and the associated global cell
number for each iteration.</td>
</tr>

<tr>
<td>MatAndGlobalCell</td>
<td>\arcanemat{MatAndGlobalCellIteratorValue}</td>
<td>\arcanemat{IMeshMaterial} <br></br> \arcanemat{MatCellVectorView}</td>
<td>Iteration over a material allowing the retrieval of the local cell number of
the material, the iteration index, and the associated global cell number for
each iteration.</td>
</tr>

<tr>
<td>AllEnvCell</td>
<td>\arcanemat{AllEnvCell}</td>
<td>\arcanemat{AllEnvCellVectorView}</td>
<td>Iteration over the AllEnvCell</td>
</tr>

<tr>
<td>EnvCell</td>
<td>\arcanemat{EnvCellLocalId}</td>
<td>\arcanemat{IMeshEnvironment} <br></br> \arcanemat{EnvCellVectorView}</td>
<td>Iteration over an environment allowing the retrieval of only the local cell
number of the environment for each iteration.</td>
</tr>

<tr>
<td>MatCell</td>
<td>\arcanemat{MatCellLocalId}</td>
<td>\arcanemat{IMeshMaterial} <br></br> \arcanemat{MatCellVectorView}</td>
<td>Iteration over a material allowing the retrieval of only the local cell
number of the material for each iteration.</td>
</tr>

</table>

If you only want to access the local numbers of material or environment cells,
it is preferable for performance reasons to use the version with `EnvCell` or
`MatCell` as the iterator type.

Here is a code example for iterating over an environment cell with the
information of the iteration index and the associated global cell.

\snippet MeshMaterialAcceleratorUnitTest.cc SampleEnvAndGlobalCell

Here is another example for iterating over the \arcanemat{AllEnvCell} and
retrieving information about the environments and materials present in each
\arcanemat{AllEnvCell}.

\snippet MeshMaterialAcceleratorUnitTest.cc SampleAllEnvCell

The class \arcanemat{ComponentCellVector} and its derived classes
(\arcanemat{MatCellVector} and \arcanemat{EnvCellVector}) are supported by the
accelerator API, and their creation or modification uses the default execution
environment.

Starting from version 3.16 of %Arcane, if the default execution environment is
an accelerator, it is possible to specify a host execution policy via the method
\arcanemat{IMeshComponent::setSpecificExecutionPolicy()}. This policy will be
used to create and modify instances of \arcanemat{ComponentCellVector},
\arcanemat{MatCellVector}, and \arcanemat{EnvCellVector}.
____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_accelerator
</span>
<span class="next_section_button">
\ref arcanedoc_acceleratorapi_reduction
</span>
</div>
