// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyNetwork.h                                        (C) 2000-2025 */
/*                                                                           */
/* Interface to handle ItemFamily relations through their connectivities.    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IITEMFAMILYNETWORK_H_ 
#define ARCANE_IITEMFAMILYNETWORK_H_ 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IItemFamily.h"
#include "arcane/core/IIncrementalItemConnectivity.h"
#include "arcane/core/IGraph2.h"

#include <functional>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemFamilyNetwork
{
 public:

  virtual ~IItemFamilyNetwork() = default;

 public:

  typedef std::function<void(IItemFamily*)> IItemFamilyNetworkTask;

 public:

  enum eSchedulingOrder
  {
    TopologicalOrder,
    InverseTopologicalOrder,
    Unknown
  };

  // TMP for debug
 public:

  static constexpr bool plug_serializer = true;

 public:

  virtual bool isActivated() const = 0;

  /*!
   * \brief Adds a dependency between two families; an element of
   * \a master_family is composed of elements of \a slave_family.
   *  The memory responsibility for \a master_to_slave_connectivity is
   *  handled by ItemFamilyNetwork
   */
  virtual void addDependency(IItemFamily* master_family, IItemFamily* slave_family,
                             IIncrementalItemConnectivity* slave_to_master_connectivity,
                             bool is_deep_connectivity = true) = 0;

  /*!
   * \brief Adds a relation between two families; an element of
   * \a source_family is connected to one or more elements of \a target_family
   *  The memory responsibility for \a source_to_target_connectivity is
   *  handled by ItemFamilyNetwork
   */
  virtual void addRelation(IItemFamily* source_family,
                           IItemFamily* target_family,
                           IIncrementalItemConnectivity* source_to_target_connectivity) = 0;

  //! Returns the dependency connectivity between the families \a source_family
  //and \a target_family
  virtual IIncrementalItemConnectivity* getDependency(IItemFamily* source_family, IItemFamily* target_family) = 0;
  virtual IIncrementalItemConnectivity* getRelation(IItemFamily* source_family, IItemFamily* target_family) = 0;

  //! Returns the connectivity between the families \a source_family and
  //\a target_family named \a name, whether it is a relation or a dependency
  virtual IIncrementalItemConnectivity* getConnectivity(IItemFamily* source_family,
                                                        IItemFamily* target_family,
                                                        const String& name) = 0;
  virtual IIncrementalItemConnectivity* getConnectivity(IItemFamily* source_family,
                                                        IItemFamily* target_family,
                                                        const String& name,
                                                        bool& is_dependency) = 0;

  /*!
   * \brief Returns, if associated with storage, the connectivity between the
   * families \a source_family and \a target_family named \a name,
   * whether it is a relation or a dependency.
   */
  virtual IIncrementalItemConnectivity* getStoredConnectivity(IItemFamily* source_family,
                                                              IItemFamily* target_family,
                                                              const String& name) = 0;
  virtual IIncrementalItemConnectivity* getStoredConnectivity(IItemFamily* source_family,
                                                              IItemFamily* target_family,
                                                              const String& name,
                                                              bool& is_dependency) = 0;

  //! Get the list of all connectivities, whether they are relations or dependencies
  virtual List<IIncrementalItemConnectivity*> getConnectivities() = 0;

  //! Get the list of all connectivities (dependencies or relations), children of a
  //family \a source_family or parents of a family \a target_family
  virtual SharedArray<IIncrementalItemConnectivity*> getChildConnectivities(IItemFamily* source_family) = 0;
  virtual SharedArray<IIncrementalItemConnectivity*> getParentConnectivities(IItemFamily* target_family) = 0;

  //! Get the list of all dependencies, children of a family \a source_family or
  //parents of a family \a target_family
  virtual SharedArray<IIncrementalItemConnectivity*> getChildDependencies(IItemFamily* source_family) = 0;
  virtual SharedArray<IIncrementalItemConnectivity*> getParentDependencies(IItemFamily* target_family) = 0;

  //! Get the list of all relations, children of a family \a source_family or parents
  //of a family \a target_family
  virtual SharedArray<IIncrementalItemConnectivity*> getChildRelations(IItemFamily* source_family) = 0;
  virtual SharedArray<IIncrementalItemConnectivity*> getParentRelations(IItemFamily* source_family) = 0;

  //! Get the list of all families
  virtual const std::set<IItemFamily*>& getFamilies() const = 0;

  virtual SharedArray<IItemFamily*> getFamilies(eSchedulingOrder order) const = 0;

  //! Schedules the execution of a task, in topological or inverse topological order
  //of the family dependency graph
  virtual void schedule(IItemFamilyNetworkTask task, eSchedulingOrder order = TopologicalOrder) = 0;

  //! Marks a connectivity as stored.
  virtual void setIsStored(IIncrementalItemConnectivity* connectivity) = 0;

  //! Retrieves information regarding the storage of the connectivity
  virtual bool isStored(IIncrementalItemConnectivity* connectivity) = 0;

  //! Retrieves information regarding the storage of the connectivity
  virtual bool isDeep(IIncrementalItemConnectivity* connectivity) = 0;

  //! Registers a graph managing DOFs connected to the mesh
  virtual Integer registerConnectedGraph(IGraph2* graph) = 0;

  //! Deregisters a graph managing DOFs connected to the mesh
  virtual void releaseConnectedGraph(Integer graph_id) = 0;

  //! Removes DOFs and links between DOFs connected to deleted cells
  virtual void removeConnectedDoFsFromCells(Int32ConstArrayView local_ids) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* IITEMFAMILYNETWORK_H_ */
