// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamily.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface of an entity family.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMFAMILY_H
#define ARCANE_CORE_IITEMFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/core/VariableTypedef.h"
#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Interface of an entity family.
 *
 * An entity family manages all entities of the same kind (Item::kind())
 * and is attached to a mesh (IMesh).
 *
 * For any mesh, there is exactly one family of
 * nodes (Node), edges (Edge), faces (Face), and cells (Cell).
 * These entities are called base mesh entities and the
 * associated families are the base mesh families.
 *
 * Depending on the implementation, there may also be families
 * of particles (Particle), dual nodes (DualNode), or links (Link).
 * Depending on the requested connectivity, a family may not have elements.
 * For example, by default in 3D, edges (Edge) are not created.
 *
 * Each entity in the family has a local identifier within the
 * family, given by Item::localId(). When a family evolves, this identifier
 * may be modified. The Item::localId() of entities in a family are not
 * necessarily contiguous. The maxLocalId() method allows knowing
 * the maximum of these values. Compaction ensures
 * that the localId() are renumbered from 0 to (nbItem()-1). For base mesh entities,
 * compaction is automatic if the mesh has the property \a "sort" set to true. For others,
 * you must call compactItems().
 *
 * By default, a family has a conversion table from
 * uniqueId() to localId(). This table must exist to allow the following operations:
 * - the uniqueId() is guaranteed to be unique within the subdomain and must
 * be so by construction across all subdomains.
 * - calling the itemsUniqueIdToLocalId() methods.
 * - the family entities can be present in multiple
 * subdomains.
 * - performing synchronizations.
 * - having partial variables on this family
 *
 * It is possible to enable or disable this conversion table
 * via the setHasUniqueIdMap() method only if no entity
 * has been created. This operation is not possible on
 * node, edge, face, and cell families.

 * When a family is modified by adding or removing entities, the
 * variables and groups relying on this family are no longer usable
 * until endUpdate() is called. For optimization reasons, it is possible to perform
 * updates of certain variables or groups via
 * partialEndUpdateVariable() or partialEndUpdateGroup(). ATTENTION, an call
 * to one of these 3 update methods invalidates the entity instances (Item).
 * To retain a reference to an entity, you must either use a group (ItemGroup)
 * or keep its unique number and use itemsUniqueIdToLocalId().
 *
 */
class ARCANE_CORE_EXPORT IItemFamily
{
  friend mesh::DynamicMesh;
  friend mesh::ItemFamily;

 public:

  virtual ~IItemFamily() {} //<! Frees resources

 public:

  virtual void build() = 0;

 public:

  //! Family name
  virtual String name() const = 0;

  //! Full family name (with the mesh's name)
  virtual String fullName() const = 0;

  //! Entity kind
  virtual eItemKind itemKind() const = 0;

  //! Number of entities
  virtual Integer nbItem() const = 0;

  /*!
   * Size required to dimension variables on these entities.
   *
   * This is the maximum of the Item::localId() of the entities in
   * this family plus 1.
   */
  virtual Int32 maxLocalId() const = 0;

 public:

  // TODO: to be removed. Use itemInfoListView instead
  //! Internal array of entities
  virtual ItemInternalArrayView itemsInternal() = 0;

 public:

  //! View on the entity information list
  virtual ItemInfoListView itemInfoListView() = 0;

  /*!
   * \brief IItemFamily parent
   *
   * Resulting from sub-mesh nesting
   * \return nullptr if there is no parent family
   */
  virtual IItemFamily* parentFamily() const = 0;

  /*!
   * \internal
   * \brief Positions the parent IItemFamily.
   *
   * To be used before build() for dynamically constructed sub-meshes
   * (i.e., not from a restart).
   *
   * TODO: To be put in the internal API
   */
  virtual void setParentFamily(IItemFamily* parent) = 0;

  //! Gives the nesting depth of the current mesh
  virtual Integer parentFamilyDepth() const = 0;

  /*!
   * \internal
   * \brief Adds a family as a dependency
   *
   * Operation symmetric to setParentFamily
   *
   * TODO: To be put in the internal API
   */
  virtual void addChildFamily(IItemFamily* family) = 0;

  //! Child families of this family
  virtual IItemFamilyCollection childFamilies() = 0;

  /*!
   * \brief Variable containing the number of the new subdomain
   * owning the entity.
   *
   * This variable is only used for mesh partitioning.
   */
  virtual VariableItemInt32& itemsNewOwner() = 0;

  //! Check the validity of internal structures (internal)
  virtual void checkValid() = 0;

  /*!
   * \brief Verification of the validity of internal structures concerning
   * connectivity.
   */
  virtual void checkValidConnectivity() = 0;

  /*!
   * \brief Checks that the \a unique_ids are truly unique
   * for all subdomains.
   *
   * This method DOES NOT check that the \a unique_ids are identical
   * to those of entities already created. It only checks the set of
   * \a unique_ids passed as arguments by all subdomains.
   *
   * This operation is collective and must be called by all subdomains.
   */
  virtual void checkUniqueIds(Int64ConstArrayView unique_ids) = 0;

 public:

  /*!
   * \brief View on the entities.
   *
   * Returns a view on the entities with local numbers \a local_ids.
   * \warning This view is only valid as long as the family does not evolve.
   * In particular, adding, removing, or compacting invalidates the view.
   * If you want to keep a list even after modification, you must
   * use groups (ItemGroup).
   */
  virtual ItemVectorView view(Int32ConstArrayView local_ids) = 0;

  /*!
   * \brief View on all entities in the family.
   */
  virtual ItemVectorView view() = 0;

  /*!
   * \brief Removes entities.
   *
   * Uses the graph (Families, Connectivities) ItemFamilyNetwork
   *
   * TODO: To be put in the internal API
   */
  virtual void removeItems2(mesh::ItemDataList& item_data_list) = 0;

  /*!
   * \internal
   * \brief Removes entities and updates connectivities.
   *
   * Does not delete any potential orphaned sub-items.
   *
   * Context of use with a family graph. Orphaned sub-items
   * must also be marked NeedRemove.
   * Therefore, there is no need to manage them in parent families.
   *
   * TODO: To be put in the internal API
   */
  virtual void removeNeedRemoveMarkedItems() = 0;

  /*!
   * \brief Unique ID entity \a unique_id.
   *
   * If no entity with this \a unique_id is found, returns \a nullptr.
   *
   * \pre hasUniqueIdMap()
   */
  ARCANE_DEPRECATED_REASON("Use MeshUtils::findOneItem() instead")
  virtual ItemInternal* findOneItem(Int64 unique_id) = 0;

  /*! \brief Notifies the end of modification of the entity list.
   *
   * This method must be called after modifying the entity list (after adding or removing). It updates the groups
   * and resizes the variables on this family.
   */
  virtual void endUpdate() = 0;

  /*!
   * \brief Partial update.
   *
   * Updates the internal structures after a family modification.
   * This is an optimized version of endUpdate() when you want
   * to perform multiple mesh modifications. This method DOES NOT update
   * the groups or variables associated with this family. Only the
   * allItems() group is available. It is possible to update
   * a group via partialEndUpdateGroup() and a variable via partialEndUpdateVariable().
   *
   * This method is reserved for experienced users. For others,
   * it is better to use endUpdate().
   */
  virtual void partialEndUpdate() = 0;

  /*!
   * \brief Updates a group.
   *
   * Updates the \a group after a family modification.
   * The update consists of removing entities from the group that were
   * possibly destroyed during the modification.
   *
   * \sa partialEndUpdate().
   */
  virtual void partialEndUpdateGroup(const ItemGroup& group) = 0;

  /*!
   * \brief Updates a variable.
   *
   * Updates the \a variable after a family modification.
   * The update consists of resizing the variable after a possible
   * addition of entities.
   *
   * \sa partialEndUpdate().
   */
  virtual void partialEndUpdateVariable(IVariable* variable) = 0;

  //! Notifies that the entities specific to the family's subdomain have been modified
  virtual void notifyItemsOwnerChanged() = 0;

  //! Notifies that the unique IDs of the entities have been modified
  virtual void notifyItemsUniqueIdChanged() = 0;

 public:

  //! Information on local connectivity within the subdomain for this family
  virtual IItemConnectivityInfo* localConnectivityInfos() const = 0;

  //! Information on global connectivity across all subdomains.
  virtual IItemConnectivityInfo* globalConnectivityInfos() const = 0;

 public:

  /*!
   * \brief Indicates whether the family has a conversion table
   * from uniqueId to localId.
   *
   * The conversion table allows using the methods
   * itemsUniqueIdToLocalId() or findOneItem().
   *
   * This method can only be called when there are no
   * entities in the family.
   *
   * The node, edge, face, and cell families of the mesh
   * must have a conversion table.
   */
  virtual void setHasUniqueIdMap(bool v) = 0;

  //! Indicates if the family has a uniqueId to localId conversion table.
  virtual bool hasUniqueIdMap() const = 0;

 public:

  /*!
   * \brief Converts an array of unique numbers to local numbers.
   *
   * This operation takes as input the \a unique_ids array containing the
   * unique numbers of entities of type \a item_kind and returns in
   * \a local_ids the corresponding local number for this subdomain.
   *
   * The complexity of this operation depends on the implementation.
   * The default implementation uses a hash table. The average complexity
   * is therefore constant.
   *
   * If \a do_fatal is true, a fatal error is generated if an entity is not
   * found, otherwise the not found element has the value NULL_ITEM_ID.
   *
   * \pre hasUniqueIdMap()
   */
  virtual void itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                                      Int64ConstArrayView unique_ids,
                                      bool do_fatal = true) const = 0;

  /*!
   * \brief Converts an array of unique numbers to local numbers.
   *
   * This operation takes as input the \a unique_ids array containing the
   * unique numbers of entities of type \a item_kind and returns in
   * \a local_ids the corresponding local number for this subdomain.
   *
   * The complexity of this operation depends on the implementation.
   * The default implementation uses a hash table. The average complexity
   * is therefore constant.
   *
   * If \a do_fatal is true, a fatal error is generated if an entity is not
   * found, otherwise the not found element has the value NULL_ITEM_ID.
   */
  virtual void itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                                      ConstArrayView<ItemUniqueId> unique_ids,
                                      bool do_fatal = true) const = 0;

 public:

  /*!
   * \brief Positions the entity sorting function.
   *
   * The default method is to sort entities by ascending uniqueId().
   * If \a sort_function is null, the default method will be used.
   * Otherwise, \a sort_function replaces the previous function, which is destroyed
   * (via delete).
   * Sorting is performed via the call to compactItems().
   * \sa itemSortFunction()
   */
  virtual void setItemSortFunction(IItemInternalSortFunction* sort_function) = 0;

  /*!
   * \brief Entity sorting function.
   *
   * The instance of this class remains the owner of the returned object,
   * which must not be destroyed or modified.
   * \sa setItemSortFunction()
   */
  virtual IItemInternalSortFunction* itemSortFunction() const = 0;

 public:

  //! Associated sub-domain
  ARCCORE_DEPRECATED_2020("Do not use this method. Try to get 'ISubDomain' from another way")
  virtual ISubDomain* subDomain() const = 0;

  //! Associated trace manager
  virtual ITraceMng* traceMng() const = 0;

  //! Associated mesh
  virtual IMesh* mesh() const = 0;

  //! Associated parallelism manager
  virtual IParallelMng* parallelMng() const = 0;

 public:

  //! Group of all entities
  virtual ItemGroup allItems() const = 0;

  //! Collection of groups in this family
  virtual ItemGroupCollection groups() const = 0;

 public:

  //! @name operations on groups
  //@{
  /*!
    \brief Searches for a group.
    \param name name of the group to search for
    \return the group named \a name or a null group if none exists.
  */
  virtual ItemGroup findGroup(const String& name) const = 0;

  /*!
   * \brief Searches for a group
   *
   * \param name name of the group to search for
   *
   * \return the found group or a null group if no group with the name
   * \a name and type \a type exists and if \a create_if_needed is false.
   * If \a create_if_needed is true, an empty group named \a name is created and returned.
   */
  virtual ItemGroup findGroup(const String& name, bool create_if_needed) = 0;

  /*!
   * \brief Creates an entity group named \a name containing the entities \a local_ids.
   *
   * \param name name of the group
   * \param local_ids list of localId() of the entities composing the group.
   * \param do_override if true and a group of the same name already exists,
   * its elements are replaced by those given in \a local_ids. If false,
   * an exception is raised.
   * \return the created group
   */
  virtual ItemGroup createGroup(const String& name, Int32ConstArrayView local_ids, bool do_override = false) = 0;

  /*!
   * \brief Creates an entity group named \a name
   *
   * The group must not already exist, otherwise an exception is raised.
   *
   * \param name name of the group
   * \return the created group
   */
  virtual ItemGroup createGroup(const String& name) = 0;

  /*!
   * \brief Deletes all groups in this family.
   */
  virtual void destroyGroups() = 0;

  /*!
   * \internal
   * For Internal Use Only
   */
  virtual ItemGroup createGroup(const String& name, const ItemGroup& parent, bool do_override = false) = 0;

  //@}

  /*!
   * \brief Searches for the variable name \a name associated with this family.
   *
   * If no variable with the name \a name exists, and if \a throw_exception is
   * false, returns 0; otherwise, it throws an exception.
   */
  virtual IVariable* findVariable(const String& name, bool throw_exception = false) = 0;

  /*!
   * \brief Adds the list of variables used by this family to the \a collection.
   */
  virtual void usedVariables(VariableCollection collection) = 0;

 public:

  //! Prepares data for dumping
  virtual void prepareForDump() = 0;

  //! Reads data from a dump
  virtual void readFromDump() = 0;

  /**
   * Copies the values of entities numbered @a source into entities
   * numbered @a destination
   *
   * @param source list of @b source localIds
   * @param destination list of @b destination localIds
   */
  virtual void copyItemsValues(Int32ConstArrayView source, Int32ConstArrayView destination) = 0;

  /**
   * Copies the mean values of entities numbered
   * @a first_source and @a second_source into entities numbered
   * @a destination
   *
   * @param first_source list of @b localIds of the 1st source
   * @param second_source list of @b localIds of the 2nd source
   * @param destination list of @b destination localIds
   */
  virtual void copyItemsMeanValues(Int32ConstArrayView first_source,
                                   Int32ConstArrayView second_source,
                                   Int32ConstArrayView destination) = 0;

  /*!
   * \brief Deletes all entities in the family.
   * \warning be careful not to destroy entities that are used in
   * by another family. In general, it is safer to use IMesh::clearItems()
   * if you want to delete all elements of the mesh.
   */
  virtual void clearItems() = 0;

  //! Compresses the entities.
  virtual void compactItems(bool do_sort) = 0;

 public:

  /*!
   * \brief Constructs the structures necessary for synchronization.
   *
   * This operation must be performed every time the entities
   * of the mesh change ownership (for example, during a load balancing).

   This operation is collective.
  */
  virtual void computeSynchronizeInfos() = 0;

  //! List of communicating sub-domains for the entities.
  virtual void getCommunicatingSubDomains(Int32Array& sub_domains) const = 0;

  //! @name variable synchronization operations
  //@{

  //! Synchronizer on all entities of the family
  virtual IVariableSynchronizer* allItemsSynchronizer() = 0;

  /*!
   * \brief Synchronizes the variables \a variables.
   *
   * The variables \a variables must all come from
   * this family and must not be partial.
   */
  virtual void synchronize(VariableCollection variables) = 0;

  // TODO: make pure virtual (December 2024)
  /*!
   * \brief Synchronizes the variables \a variables on a list of entities.
   *
   * The variables \a variables must all come from
   * this family and must not be partial.
   *
   * Only the entities listed in \a local_ids will be synchronized. Note:
   * an entity present in this list on one sub-domain must be present
   * in this list for any other sub-domain that possesses this entity.
   */
  virtual void synchronize(VariableCollection variables, Int32ConstArrayView local_ids);
  //@}

  /*!
   * \brief Applies a reduction operation from ghost items.
   *
   * This operation is the inverse of synchronization.
   *
   * The sub-domain retrieves the values of variable \a v on the entities
   * it shares with other sub-domains, and the reduction operation
   * \a operation is applied to this variable.
   */
  virtual void reduceFromGhostItems(IVariable* v, IDataOperation* operation) = 0;
  /*!
   * \brief Applies a reduction operation from ghost items.
   *
   * This operation is the inverse of synchronization.
   *
   * The sub-domain retrieves the values of variable \a v on the entities
   * it shares with other sub-domains, and the reduction operation
   * \a operation is applied to this variable.
   */
  virtual void reduceFromGhostItems(IVariable* v, Parallel::eReduceType operation) = 0;

  //! Searches for an adjacency list.
  ARCANE_DEPRECATED_REASON("Y2024: use findAdjacency() instead")
  virtual ItemPairGroup findAdjencyItems(const ItemGroup& group,
                                         const ItemGroup& sub_group,
                                         eItemKind link_kind,
                                         Integer nb_layer) = 0;
  /*!
   * \brief Searches for an adjacency list.
   *
   * Searches for the list of entities of type \a sub_kind, linked by
   * the entity type \a link_kind of group \a group,
   * over a number of layers \a nb_layer.
   *
   * If \a group and \a sub_group are of the same kind, an entity is always
   * in its adjacency list, as the first element.
   *
   * If the list does not exist, it is created.
   *
   * \note currently only one layer is allowed.
   */
  virtual ItemPairGroup findAdjacencyItems(const ItemGroup& group,
                                           const ItemGroup& sub_group,
                                           eItemKind link_kind,
                                           Integer nb_layer);

  /*!
   * \brief Returns the interface of the particle family for this family.
   *
   * The IParticleFamily interface only exists if this family is
   * a particle family (itemKind()==IK_Particle). For other family kinds,
   * 0 is returned.
   */
  virtual IParticleFamily* toParticleFamily() = 0;

  /*!
   * \brief Returns the interface of the particle family for this family.
   *
   * The IParticleFamily interface only exists if this family is
   * a particle family (itemKind()==IK_Particle). For other family kinds, 0 is returned.
   */
  virtual IDoFFamily* toDoFFamily() { return nullptr; }

  /*!
   * \internal
   * \brief Removes the entities given by \a local_ids.
   *
   * For internal use only. If you want to delete entities
   * from the mesh, you must go through IMeshModifier via the call to IMesh::modifier().
   */
  virtual void internalRemoveItems(Int32ConstArrayView local_ids, bool keep_ghost = false) = 0;

  /*!
   * \name Register/Delete a connectivity manager.
   *
   * Allows propagating family changes to "external" connectivities
   * in which it is involved.
   * These "external" connectivities are currently those
   * using degrees of freedom.
   *
   * \note These methods are internal to %Arcane.
   */
  //@{
  virtual void addSourceConnectivity(IItemConnectivity* connectivity) = 0;
  virtual void addTargetConnectivity(IItemConnectivity* connectivity) = 0;
  virtual void removeSourceConnectivity(IItemConnectivity* connectivity) = 0;
  virtual void removeTargetConnectivity(IItemConnectivity* connectivity) = 0;
  virtual void setConnectivityMng(IItemConnectivityMng* connectivity_mng) = 0;
  //@}

  /*!
    * \brief Allocates ghost entities.
    *
    * After calling this operation, you must call endUpdate() to
    * notify the instance that the modifications are finished. It is possible
    * to chain several allocations before calling
    * endUpdate().
    *
    * The \a unique_ids are those of items present on another
    * sub-domain, whose number is in the owners array (of the same
    * size as the unique_ids array). \a items must have the same
    * number of elements as \a unique_ids and will be filled back
    * with the local numbers of the created entities.
    */
  virtual void addGhostItems(Int64ConstArrayView unique_ids, Int32ArrayView items,
                             Int32ConstArrayView owners) = 0;

 public:

  //! Interface of behaviors/policies associated with this family.
  virtual IItemFamilyPolicyMng* policyMng() = 0;

  //! Properties associated with this family.
  virtual Properties* properties() = 0;

 public:

  //! Event for entity addition and deletion
  virtual EventObservableView<const ItemFamilyItemListChangedEventArgs&> itemListChangedEvent() = 0;

 public:

  /*!
   * \brief Changes the unique number of the entity.
   *
   * \warning This method is experimental.
   * \warning Modifying an entity's uniqueId can cause inconsistencies
   * in the mesh and numbering. It is preferable to only call this method
   * on entities that are not associated with others (for example, nodes that
   * have just been created).
   */
  virtual void experimentalChangeUniqueId(ItemLocalId local_id, ItemUniqueId unique_id) = 0;

 public:

  /*!
   * \brief Resizes the variables of this family.
   *
   * This method is internal to Arcane.
   */
  virtual void resizeVariables(bool force_resize) = 0;

 public:

  //! Topology modifier interface.
  virtual IItemFamilyTopologyModifier* _topologyModifier() = 0;

 public:

  //! Internal Arcane API
  virtual IItemFamilyInternal* _internalApi() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
