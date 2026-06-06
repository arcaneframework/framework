// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshUtils.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Various utility functions for the mesh.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHUTILS_H
#define ARCANE_CORE_MESHUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FunctorUtils.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file MeshUtils.h
 *
 * \brief Utility functions for the mesh.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class XmlNode;
class IVariableSynchronizer;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MeshUtils
{
extern "C++" ARCANE_CORE_EXPORT void
writeMeshItemInfo(ISubDomain*, Cell cell, bool depend_info = true);
extern "C++" ARCANE_CORE_EXPORT void
writeMeshItemInfo(ISubDomain*, Node node, bool depend_info = true);
extern "C++" ARCANE_CORE_EXPORT void
writeMeshItemInfo(ISubDomain*, Edge edge, bool depend_info = true);
extern "C++" ARCANE_CORE_EXPORT void
writeMeshItemInfo(ISubDomain*, Face face, bool depend_info = true);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Reorders the nodes of a face.
 *
 * This method reorders the list of nodes of a face so that the following
 * properties are respected:
 * - the first node of the face is the one with the smallest global number.
 * - the second node of the face is the one with the second smallest global number.
 *
 * This allows faces to be oriented identically in parallel.
 *
 * \a before_ids and \a to must have the same number of elements
 *
 * \param before_ids global numbers of the face nodes before renumbering.
 * \param after_ids (output), global numbers of the face nodes after renumbering
 *
 * \retval true if the face changes orientation during renumbering
 * \retval false otherwise.
 */
extern "C++" ARCANE_CORE_EXPORT bool
reorderNodesOfFace(Int64ConstArrayView before_ids, Int64ArrayView after_ids);

extern "C++" ARCANE_CORE_EXPORT bool
reorderNodesOfFace(Int64ConstArrayView before_ids, Int64ArrayView after_ids);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Reorders the nodes of a face.
 *
 * This method reorders the list of nodes of a face so that the following
 * properties are respected:
 * - the first node of the face is the one with the smallest global number.
 * - the second node of the face is the one with the second smallest global number.
 *
 * This allows faces to be oriented identically in parallel.
 *
 * \a nodes_unique_id and \a new_index must have the same number of elements
 *
 * \param nodes_unique_id unique numbers of the face nodes.
 * \param new_index (output), position of the node numbers after reorientation.
 *
 * For example, if a face has 4 nodes with unique numbers 7 3 2 5,
 * the reorientation will yield the quadruplet (2 3 7 5), which is the following index array
 * (2,1,0,3).
 *
 * \retval true if the face changes orientation during renumbering
 * \retval false otherwise.
 */
extern "C++" ARCANE_CORE_EXPORT bool
reorderNodesOfFace2(Int64ConstArrayView nodes_unique_id, Int32ArrayView new_index);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Searches for a face entity using the local numbers of these nodes.
 *
 * Searches for the face given by the sorted list of <b>local</b> numbers of
 * these nodes \a face_nodes_local_id. \a node must be the first node of
 * the face. The nodes of the face must be correctly oriented, as
 * after calling reorderNodesOfFace().
 *
 * \return the corresponding face or a null face if it is not found.
 */
extern "C++" ARCANE_CORE_EXPORT Face
getFaceFromNodesLocalId(Node node, Int32ConstArrayView face_nodes_local_id);

ARCANE_DEPRECATED_REASON("Y2025: Use getFaceFromNodesLocalId() instead")
inline Face
getFaceFromNodesLocal(Node node, Int32ConstArrayView face_nodes_local_id)
{
  return getFaceFromNodesLocalId(node, face_nodes_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Searches for a face entity using the unique numbers of these nodes.
 *
 * Searches for the face given by the sorted list of <b>unique</b> numbers of
 * these nodes \a face_nodes_unique_id. \a node must be the first node of
 * the face. The nodes of the face must be correctly oriented, as
 * after calling reorderNodesOfFace().
 *
 * \return the corresponding face or a null face if it is not found.
 */
extern "C++" ARCANE_CORE_EXPORT Face
getFaceFromNodesUniqueId(Node node, Int64ConstArrayView face_nodes_unique_id);

ARCANE_DEPRECATED_REASON("Y2025: Use getFaceFromNodesUniqueId() instead")
inline Face
getFaceFromNodesUnique(Node node, Int64ConstArrayView face_nodes_unique_id)
{
  return getFaceFromNodesUniqueId(node, face_nodes_unique_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Generates a unique identifier from a list of node identifiers.
 */
extern "C++" ARCANE_CORE_EXPORT Int64
generateHashUniqueId(SmallSpan<const Int64> nodes_unique_id);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Removes an entity while preserving order.
 *
 * Removes the entity with local number \a local_id from the list \a items.
 * Entities located after the removed entity are shifted to fill the gap.
 * If no value in \a items equals \a local_id, an exception is raised.
 */
extern "C++" ARCANE_CORE_EXPORT void
removeItemAndKeepOrder(Int32ArrayView items, Int32 local_id);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Checks if the mesh has certain properties.
 *
 * If \a is_sorted, checks that the mesh entities are sorted by
 * ascending order of their uniqueId().
 * If \a has_no_hole is true, checks that if the mesh has \a n
 * entities of a certain type, their local number ranges from \a 0 to \a n-1.
 * If \a check_faces is true, it checks the faces. This option
 * is only usable for older generation meshes (MeshV1) and
 * will be removed as soon as this option is no longer used.
 */
extern "C++" ARCANE_CORE_EXPORT void
checkMeshProperties(IMesh* mesh, bool is_sorted, bool has_no_hole, bool check_faces);

/*!
 * \brief Writes the mesh info \a mesh to the file \a file_name
 *
 * Entity identifiers are sorted so that the mesh is identical regardless of the initial numbering.
 */
extern "C++" ARCANE_CORE_EXPORT void
writeMeshInfosSorted(IMesh* mesh, const String& file_name);

extern "C++" ARCANE_CORE_EXPORT void
writeMeshInfos(IMesh* mesh, const String& file_name);

/*!
 * \brief Writes the connectivity of the mesh \a mesh to the file \a file_name
 *
 * The connectivity of each edge, face, and cell entity is saved.
 */
extern "C++" ARCANE_CORE_EXPORT void
writeMeshConnectivity(IMesh* mesh, const String& file_name);

extern "C++" ARCANE_CORE_EXPORT void
checkMeshConnectivity(IMesh* mesh, const XmlNode& root_node, bool check_sub_domain);

extern "C++" ARCANE_CORE_EXPORT void
checkMeshConnectivity(IMesh* mesh, const String& file_name, bool check_sub_domain);

/*!
 * \brief Writes the description of items in group \a item_group to the stream \a ostr
 *
 * For display, a name \a name is associated.
 */
extern "C++" ARCANE_CORE_EXPORT void
printItems(std::ostream& ostr, const String& name, ItemGroup item_group);

/*!
 * \brief Displays the memory usage of the mesh groups.
 *
 * If \a print_level is 0, only the total memory usage is displayed.
 * If \a print_level is 1 or more, the usage for each group is displayed.
 *
 * Returns the memory consumed in bytes.
 */
extern "C++" ARCANE_CORE_EXPORT Int64
printMeshGroupsMemoryUsage(IMesh* mesh, Int32 print_level);

//! Optimizes the memory usage of the groups.
extern "C++" ARCANE_CORE_EXPORT void
shrinkMeshGroups(IMesh* mesh);

/*!
 * \brief Writes the topology information of a synchronizer to a file
 *
 * Writes the topology information of \a var_syncer to the file \a filename.
 * This method is collective. Only rank 0 writes the topology information.
 */
extern "C++" ARCANE_CORE_EXPORT void
dumpSynchronizerTopologyJSON(IVariableSynchronizer* var_syncer, const String& filename);

/*!
 * \internal
 * \brief Calculates and displays common patterns in the connectivities.
 */
extern "C++" ARCANE_CORE_EXPORT void
computeConnectivityPatternOccurence(IMesh* mesh);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Indicates that the mesh connectivities will not be regularly modified.
 *
 * This function allows indicating that the connectivities associated with
 * mesh entities (Node, Edge, Face, and Cell) are mostly read-only. Note that
 * this does not concern particles.
 *
 * When used on an accelerator, this allows duplicating information between
 * the accelerator and the host to avoid multiple round trips
 * if the connectivities are used on both simultaneously.
 *
 * If \a q is non-null and \a do_prefetch is true, then
 * VariableUtils::prefetchVariableAsync() is called for each variable
 * managing the connectivity.
 */
extern "C++" ARCANE_CORE_EXPORT void
markMeshConnectivitiesAsMostlyReadOnly(IMesh* mesh, RunQueue* q = nullptr,
                                       bool do_prefetch = false);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * \brief Returns the entity of the family \a family with unique ID \a unique_id.
 *
 * If no entity with this \a unique_id is found, returns the null entity.
 *
 * \pre family->hasUniqueIdMap() == true
 */
extern "C++" ARCANE_CORE_EXPORT ItemBase
findOneItem(IItemFamily* family, Int64 unique_id);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * \brief Returns the entity of the family \a family with unique ID \a unique_id.
 *
 * If no entity with this \a unique_id is found, returns the null entity.
 *
 * \pre family->hasUniqueIdMap() == true
 */
extern "C++" ARCANE_CORE_EXPORT ItemBase
findOneItem(IItemFamily* family, ItemUniqueId unique_id);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Visits all groups of \a family using the functor \a functor.
 */
extern "C++" ARCANE_CORE_EXPORT void
visitGroups(IItemFamily* family, IFunctorWithArgumentT<ItemGroup&>* functor);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Visits all groups of \a mesh using the functor \a functor.
 */
extern "C++" ARCANE_CORE_EXPORT void
visitGroups(IMesh* mesh, IFunctorWithArgumentT<ItemGroup&>* functor);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Visits all groups of \a family using the lambda \a f.
 *
 * This function allows applying a visitor to all groups of the family \a family.
 *
 * For example:
 *
 * \code
 * IMesh* mesh = ...;
 * auto xx = [](const ItemGroup& x) { std::cout << "name=" << x.name(); };
 * MeshUtils::visitGroups(mesh,xx);
 * \endcode
 */
template <typename LambdaType> inline void
visitGroups(IItemFamily* family, const LambdaType& f)
{
  StdFunctorWithArgumentT<ItemGroup&> sf(f);
  // It must be cast to the correct type so that the compiler uses the correct overload.
  IFunctorWithArgumentT<ItemGroup&>* sf_addr = &sf;
  visitGroups(family, sf_addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Visits all groups of \a mesh using the lambda \a f.
 *
 * This function allows applying a visitor to all groups of the families in
 * the mesh \a mesh
 *
 * It is used as follows:
 *
 * \code
 * IMesh* mesh = ...;
 * auto xx = [](const ItemGroup& x) { std::cout << "name=" << x.name(); };
 * MeshVisitor::visitGroups(mesh,xx);
 * \endcode
 */
template <typename LambdaType> inline void
visitGroups(IMesh* mesh, const LambdaType& f)
{
  StdFunctorWithArgumentT<ItemGroup&> sf(f);
  // It must be cast to the correct type so that the compiler uses the correct overload.
  IFunctorWithArgumentT<ItemGroup&>* sf_addr = &sf;
  visitGroups(mesh, sf_addr);
}

namespace impl
{
  inline Int64 computeCapacity(Int64 size)
  {
    return Arcane::MemoryUtils::impl::computeCapacity(size);
  }
} // namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Resizes an array that is indexed by 'ItemLocalId'.
 *
 * The array \a array is resized only if \a new_size is greater than the
 * current size of the array or if \a force_resize is true.
 *
 * If the array is resized, additional capacity is reserved to avoid
 * reallocating every time.
 *
 * This function is generally called for arrays indexed by an ItemLocalId,
 * and therefore this function can be called every time an entity is added
 * to the mesh.
 *
 * \retval true if a resize occurred
 * \retval false otherwise
 */
template <typename DataType> inline bool
checkResizeArray(Array<DataType>& array, Int64 new_size, bool force_resize)
{
  return Arcane::MemoryUtils::checkResizeArrayWithCapacity(array, new_size, force_resize);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Returns the maximum of the uniqueId() of the standard entities of the mesh.
 *
 * Standard entities are nodes, cells, faces, and edges.
 * The operation is collective on mesh->parallelMng().
 */
extern "C++" ARCANE_CORE_EXPORT ItemUniqueId
getMaxItemUniqueIdCollective(IMesh* mesh);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Checks the hash of the uniqueId() of entities in a family.
 *
 * Calculates a hash of the uniqueId() of entities in a family using the
 * \a hash_algo algorithm. For this calculation, rank 0 retrieves the set of
 * uniqueIds() of the local entities of each subdomain, sorts them, and
 * calculates the hash on the sorted array.
 *
 * Since most of the work is performed by rank 0, this method is not very
 * extensible and should therefore only be used for testing purposes.
 *
 * \a expected_hash is the expected hash value in hexadecimal characters
 * (obtained via Convert::toHexaString()). If \a expected_hash is not null,
 * it compares the result with this value and, if different, throws a
 * FatalErrorException.
 *
 * This operation is collective.
 */
extern "C++" ARCANE_CORE_EXPORT void
checkUniqueIdsHashCollective(IItemFamily* family, IHashAlgorithm* hash_algo,
                             const String& expected_hash, bool print_hash_value,
                             bool include_ghost);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fills \a uids with the uniqueId() of the entities in \a view.
 */
extern "C++" ARCANE_CORE_EXPORT void
fillUniqueIds(ItemVectorView items,Array<Int64>& uids);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates or recreates a node-node connectivity via edges.
 *
 * The connectivity will be named \a connectivity_name.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IIndexedIncrementalItemConnectivity>
computeNodeNodeViaEdgeConnectivity(IMesh* mesh, const String& connectivity_name);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates or recreates a node-node connectivity via edges for nodes
 * on the boundary faces of the mesh.
 *
 * The connectivity will be named \a connectivity_name.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IIndexedIncrementalItemConnectivity>
computeBoundaryNodeNodeViaEdgeConnectivity(IMesh* mesh, const String& connectivity_name);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Determines the owners of the nodes.
 *
 * The determination is based on the owners of the cells.
 * There should be no layers of ghost cells.
 *
 * This operation is collective.
 */
extern "C++" ARCANE_CORE_EXPORT void
computeAndSetOwnerForNodes(IMesh* mesh);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Determines the owners of the edges.
 *
 * The determination is based on the owners of the cells.
 * There should be no layers of ghost cells.
 *
 * This operation is collective.
 */
extern "C++" ARCANE_CORE_EXPORT void
computeAndSetOwnerForEdges(IMesh* mesh);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Determines the owners of the faces.
 *
 * The determination is based on the owners of the cells.
 * There should be no layers of ghost cells.
 *
 * This operation is collective.
 */
extern "C++" ARCANE_CORE_EXPORT void
computeAndSetOwnerForFaces(IMesh* mesh);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MeshUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh_utils
{
// Used for compatibility with existing code.
// These usings were added for Arcane version 3.10 (June 2023).
// They can be deprecated starting in early 2024.
using MeshUtils::checkMeshConnectivity;
using MeshUtils::checkMeshProperties;
using MeshUtils::computeConnectivityPatternOccurence;
using MeshUtils::dumpSynchronizerTopologyJSON;
using MeshUtils::getFaceFromNodesLocal;
using MeshUtils::getFaceFromNodesUnique;
using MeshUtils::printItems;
using MeshUtils::printMeshGroupsMemoryUsage;
using MeshUtils::removeItemAndKeepOrder;
using MeshUtils::reorderNodesOfFace;
using MeshUtils::reorderNodesOfFace2;
using MeshUtils::shrinkMeshGroups;
using MeshUtils::writeMeshConnectivity;
using MeshUtils::writeMeshInfos;
using MeshUtils::writeMeshInfosSorted;
using MeshUtils::writeMeshItemInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh_utils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
