// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyExchanger.h                                      (C) 2000-2025 */
/*                                                                           */
/* Exchange of family entities between sub-domains.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMFAMILYEXCHANGER_H
#define ARCANE_CORE_IITEMFAMILYEXCHANGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/VariableCollection.h"

#include "arcane/mesh/MeshGlobal.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParallelExchangerOptions;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Exchange of entities and their characteristics for a given family
 
 This class manages the exchange of entities between sub-domains. It is
 used, for example, during a redistribution. Generally, this class
 is not used directly (except to specify the entities to exchange)
 but via the IMeshExchanger interface.
 
 The user of this class must start by specifying the list of entities to
 send to each sub-domain via the setExchangeItems() method.

 Entity exchange is done in several steps as indicated in IMeshExchanger.

 The actual serialization of entities takes place in three successive phases:
 entities, groups, and variables. Deserialization is done in the same order.
 Indeed, it is necessary to know the groups to deserialize the variables, and
 to know the entities to deserialize the groups.

 When meshes or particles are sent, you must call the readAndAllocItems()
 method to create them, before calling readGroups() and then readVariables().
*/
class ARCANE_CORE_EXPORT IItemFamilyExchanger
{
 public:

  virtual ~IItemFamilyExchanger(){}

 public:

  /*!
   * \internal
   * \brief Determines the list of entities to exchange.

   * \warning This method must only be used for particle families.

   This operation uses the itemsOwner() variable and the owner() field of each
   entity to determine who each entity must be sent to. Therefore, this
   operation must be called before DynamicMesh::_setOwnerFromVariable() is
   called.
   *
   * \todo To be removed
   */
  virtual void computeExchangeItems() =0;
  
  //! Positions the list of entities to exchange.
  virtual void setExchangeItems(ConstArrayView< std::set<Int32> > items_to_send) =0;

  /*!
   * \brief Determines the information necessary for the exchanges.
   * \retval true if there is nothing to exchange
   * \retval false otherwise.
   */
  virtual bool computeExchangeInfos() =0;

  //! Prepares the sending structures
  virtual void prepareToSend() =0;
  virtual void releaseBuffer() =0;

  /*!
   * \brief After receiving messages, reads and creates the transferred entities.
   *
   * This method does nothing for entities other than meshes and particles,
   * for legacy management.
   * With the ItemFamilyNetwork family graph, this method creates the items and
   * their dependencies (i.e., descendant connectivities).
   * This involves separating the processing of sub-items (sub-meshes) and
   * relations (ascending connectivities or dofs), which cannot be processed
   * until all items are created.
   *
   * \warning Before calling this method, you must be certain that entities no longer
   * belonging to this sub-domain have been destroyed
   */
  virtual void readAndAllocItems() =0;
  virtual void readAndAllocSubMeshItems() =0;
  virtual void readAndAllocItemRelations() =0;

  //! After receiving messages, reads the groups
  virtual void readGroups() =0;

  //! After receiving messages, reads the variable values
  virtual void readVariables() =0;

  /*!
   * \internal
   * \brief Removes the sent entities.
   *
   * This operation must only be performed for entities that do not depend on
   * another entity. For example, it is impossible to directly delete nodes,
   * because certain meshes that are not sent may rely on them.
   *
   * \warning This operation is only valid for particles without the concept of
   * ghost particles.
   * \todo To be removed
   */
  virtual void removeSentItems() =0;

  //! Sends the exchange messages
  virtual void processExchange() =0;

  /*!
   * \brief Finalizes the exchange.
   *
   * Performs the final updates following an exchange. This method is called
   * when all entities and variables have been exchanged.
   */
  virtual void finalizeExchange() =0;

  //! Associated family
  virtual IItemFamily* itemFamily() =0;

  //! Sets the options used during entity exchange
  virtual void setParallelExchangerOption(const ParallelExchangerOptions& options) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
