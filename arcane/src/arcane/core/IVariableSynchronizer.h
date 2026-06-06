// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableSynchronizer.h                                     (C) 2000-2024 */
/*                                                                           */
/* Interface of a variable synchronization service.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLESYNCHRONIZER_H
#define ARCANE_CORE_IVARIABLESYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a variable synchronization service.
 *
 * This class is managed by Arcane and is generally not necessary
 * to use directly. If you want to synchronize a variable,
 * you just need to call the VariableRef::synchronize() method.
 *
 * An instance of this class is created via
 * IParallelMng::createVariableSynchronizer(). An instance is associated
 * with an entity group. You must call the compute() function
 * to calculate the synchronization information. If the entities are
 * compacted, you must call changeLocalIds().
 */
class ARCANE_CORE_EXPORT IVariableSynchronizer
{
 public:

  virtual ~IVariableSynchronizer() {}

 public:

  //! Associated parallel manager
  virtual IParallelMng* parallelMng() = 0;

  /*!
   * \brief Entity group used for synchronization.
   *
   * The current implementation only supports the group
   * of all entities of a family.
   */
  virtual const ItemGroup& itemGroup() = 0;

  /*!
   * \brief Recalculates the synchronization information.
   *
   * This operation is collective.
   *
   * This function must be called if the entities in itemGroup() change
   * owner or if the group itself evolves.
   * TODO: call this function automatically if needed.
   */
  virtual void compute() = 0;

  //! Called when the local IDs of the entities are modified.
  virtual void changeLocalIds(Int32ConstArrayView old_to_new_ids) = 0;

  //! Synchronizes the variable \a var in blocking mode
  virtual void synchronize(IVariable* var) = 0;

  // TODO: make pure virtual (December 2024)
  /*!
   * \brief Synchronizes the variable \a var on the entities \a local_ids in blocking mode
   * 
   * Only the entities listed in \a local_ids will be synchronized. Note:
   * an entity present in this list on one subdomain must be present
   * in this list for any other subdomain that owns this entity.
   */
  virtual void synchronize(IVariable* var, Int32ConstArrayView local_ids);
  
  /*!
   * \brief Synchronizes the variables \a vars in blocking mode.
   *
   * All variables must belong to the same family
   * and this entity group.
   */
  virtual void synchronize(VariableCollection vars) = 0;

  // TODO: make pure virtual (December 2024)
  /*!
   * \brief Synchronizes the variables \a vars in blocking mode.
   *
   * All variables must belong to the same family
   * and this entity group.
   * 
   * Only the entities listed in \a local_ids will be synchronized. Note:
   * an entity present in this list on one subdomain must be present
   * in this list for any other subdomain that owns this entity.
   */
  virtual void synchronize(VariableCollection vars, Int32ConstArrayView local_ids);
  
  /*!
   * \brief Ranks of subdomains with which communication occurs.
   */
  virtual Int32ConstArrayView communicatingRanks() = 0;

  /*!
   * \brief List of local IDs of entities shared with a subdomain.
   *
   * The rank of the subdomain is that of communicatingRanks()[index].
   */
  virtual Int32ConstArrayView sharedItems(Int32 index) = 0;

  /*!
   * \brief List of local IDs of ghost entities with a subdomain.
   *
   * The rank of the subdomain is that of communicatingRanks()[index].
   */
  virtual Int32ConstArrayView ghostItems(Int32 index) = 0;

  /*!
   * \brief Synchronizes the data \a data.
   *
   * The data \a data must be associated with a variable for which
   * it is valid to call \a synchronize(). This method is internal
   * to Arcane.
   */
  virtual void synchronizeData(IData* data) = 0;

  /*!
   * \brief Event sent at the beginning and end of synchronization.
   *
   * This event is sent during calls to the methods
   * synchronize(IVariable* var)
   * and synchronize(VariableCollection vars). If you wish to be notified
   * of synchronizations for all instances of IVariableSynchronizer,
   * you must use IVariableMng::synchronizerMng().
   */
  virtual EventObservable<const VariableSynchronizerEventArgs&>& onSynchronized() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
