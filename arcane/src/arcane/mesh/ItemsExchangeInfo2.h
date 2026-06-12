// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemsExchangeInfo2.h                                        (C) 2000-2024 */
/*                                                                           */
/* Information for exchanging entities and their characteristics.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMSEXCHANGEINFO2_H
#define ARCANE_MESH_ITEMSEXCHANGEINFO2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/VariableCollection.h"
#include "arcane/core/IItemFamilyExchanger.h"
#include "arcane/core/IItemFamilySerializeStep.h"
#include "arcane/core/ParallelExchangerOptions.h"

#include "arcane/mesh/MeshGlobal.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemInternal;
class IParallelExchanger;
class IItemFamilySerializer;
class IItemFamilySerializeStep;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class ItemGroupsSerializer2;
class TiedInterfaceExchanger;
class ItemFamilyVariableSerializer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Information for exchanging entities of a given family
 * and their characteristics.

 An instance of this class contains all the information to
 exchange the mesh entities \a m_mesh linked to the family \a item_family.

 The exchange of entities behaves differently depending on the kind (eItemKind)
 of the entity. For cells, the complete description of the connectivity
 is sent to the receiving sub-domain. For nodes (Node), edges (Edge)
 and faces (Face), the connectivity is not sent because it is given
 by the cells. It is therefore not possible to serialize these three
 types of entities independently of the cells (which would not be consistent
 anyway). For particles, the cell number to which each particle belongs is
 also sent.

 When cells or particles are sent, it is necessary
 to call the readAndAllocItems() method to create them, before
 calling readGroups() then readVariables().

 In addition to the entities themselves, this class exchanges the values of
 variables as well as the group memberships.
*/
class ARCANE_MESH_EXPORT ItemsExchangeInfo2
: public TraceAccessor
, public IItemFamilyExchanger
{
 public:

  ItemsExchangeInfo2(IItemFamily* item_family);
  ~ItemsExchangeInfo2();

 public:

  void computeExchangeItems() override;

  void setExchangeItems(ConstArrayView<std::set<Int32>> items_to_send) override;

  /*!
   * \brief Determines the necessary information for the exchanges.
   * \retval true if there is nothing to exchange
   * \retval false otherwise.
   */
  bool computeExchangeInfos() override;

  //! Prepares the sending structures
  void prepareToSend() override;
  void releaseBuffer() override;

  /*!
   * \brief After receiving messages, reads and creates the transferred entities.
   *
   * This method does nothing for entities other
   * than cells and particles.
   *
   * \warning Before calling this method, it must be certain
   * that the entities no longer belonging to this sub-domain have been
   * destroyed
   */
  void readAndAllocItems() override;
  void readAndAllocSubMeshItems() override;
  void readAndAllocItemRelations() override;

  //! After receiving messages, reads the groups
  void readGroups() override;

  //! After receiving messages, reads the variable values
  void readVariables() override;

  /*!
   * \brief Deletes the sent entities.
   *
   * This operation must only be performed for entities which
   * do not depend on another entity. For example, it is impossible
   * to directly delete nodes, because some cells which
   * are not sent may rely on them.
   *
   * In practice, this operation is only useful for particles.
   */
  void removeSentItems() override;

  //! Sends the exchange messages
  void processExchange() override;

  /*!
   * \brief Finalizes the exchange.
   *
   * Performs the last updates following an exchange. This
   * method is called when all entities and variables
   * have been exchanged.
   */
  void finalizeExchange() override;

  IItemFamily* itemFamily() override { return m_item_family; }

  void setParallelExchangerOption(const ParallelExchangerOptions& options) override;

 public:

  void addSerializeStep(IItemFamilySerializeStep* step);

 private:

  IItemFamily* m_item_family;

  //! List of entities to send to each processor
  UniqueArray<SharedArray<Int32>> m_send_local_ids;

  //! Serializer of groups
  UniqueArray<ItemGroupsSerializer2*> m_groups_serializers;

  /*!
   * \brief List of families included in the exchange.
   *
   * It consists of \a m_item_family and these child families
   * (at a single level).
   */
  UniqueArray<IItemFamily*> m_families_to_exchange;

  Ref<IParallelExchanger> m_exchanger;

  /*!
   * \brief List of local IDs of received entities.
   */
  UniqueArray<SharedArray<Int32>> m_receive_local_ids;

  IItemFamilySerializer* m_family_serializer;

  UniqueArray<IItemFamilySerializeStep*> m_serialize_steps;

  ParallelExchangerOptions m_exchanger_option;

 private:

  inline void _addItemToSend(Int32 sub_domain_id, Item item);
  bool _computeExchangeInfos();
  void _applySerializeStep(IItemFamilySerializeStep::ePhase phase,
                           const ItemFamilySerializeArgs& args);
  void _applyDeserializePhase(IItemFamilySerializeStep::ePhase phase);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
