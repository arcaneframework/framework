// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshExchanger.h                                            (C) 2000-2022 */
/*                                                                           */
/* Mesh exchange management between subdomains.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESHEXCHANGER_H
#define ARCANE_IMESHEXCHANGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IPrimaryMesh;
class IItemFamily;
class IItemFamilyExchanger;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Management of a mesh exchange between subdomains.
 *
 * An exchange takes place in several phases, which must be executed
 * in the order dictated by the ePhase enumeration.
 *
 */
class ARCANE_CORE_EXPORT IMeshExchanger
{
 public:

  //! Indicates the different phases of the exchange
  enum class ePhase
  {
    Init = 0,
    ComputeInfos,
    ProcessExchange,
    RemoveItems,
    AllocateItems,
    UpdateItemGroups,
    UpdateVariables,
    Finalize,
    Ended
  };

 public:

  virtual ~IMeshExchanger() {} //<! Releases resources

 public:

  /*!
   * \brief Calculates the information to send/receive from other subdomains.
   *
   * This operation is collective.
   *
   * The calculation of information to send is done by knowing the new
   * owner of each entity. This information is stored in
   * the IItemFamily::itemsNewOwner() variable. For example, a mesh
   * will be migrated if the new owner is different from the current owner
   * (which is given by Item::owner()).
   *
   * After calling this method, each mesh entity is modified as follows:
   * - the Item::owner() field indicates the new owner.
   * - the entities that will be deleted after the exchange are marked by the flag
   *   ItemFlags::II_NeedRemove (except for now for particles
   *   without the concept of ghosts, but this is temporary).
   *
   * Returns true if there is no exchange to perform.
   *
   * \pre phase()==ePhase::ComputeInfos
   * \post phase()==ePhase::ProcessExchange
   */
  virtual bool computeExchangeInfos() = 0;

  /*!
   * \brief Performs the exchange of information between subdomains.
   *
   * This operation is collective.
   *
   * This operation makes no modification to the mesh. It simply
   * sends and receives the necessary information for
   * mesh update.
   *
   * \pre phase()==ePhase::ProcessExchange
   * \post phase()==ePhase::RemoveItems
   */
  virtual void processExchange() = 0;

  /*!
   * \brief Deletes from this subdomain the entities that should no longer
   * be there following the exchange.
   *
   * All entities marked with the ItemFlags::II_NeedRemove flag
   * are deleted.
   *
   * \pre phase()==ePhase::RemoveItems
   * \post phase()==ePhase::AllocateItems
   */
  virtual void removeNeededItems() = 0;

  /*!
   * \brief Allocates the entities received from other subdomains.
   *
   * This operation is collective.
   *
   * \pre phase()==ePhase::AllocateItems
   * \post phase()==ePhase::UpdateItemGroups
   */
  virtual void allocateReceivedItems() = 0;

  /*!
   * \brief Update of entity groups
   *
   * This operation is collective.
   *
   * \pre phase()==ePhase::UpdateItemGroups
   * \post phase()==ePhase::UpdateVariables
   */
  virtual void updateItemGroups() = 0;

  /*!
   * \brief Update of variables
   *
   * This operation is collective.
   *
   * \pre phase()==ePhase::UpdateVariables
   * \post phase()==ePhase::Finalize
   */
  virtual void updateVariables() = 0;

  /*!
   * \brief Finalizes the exchanges.
   *
   * This operation is collective.
   *
   * This method performs the last necessary operations during
   * the exchange.
   *
   * \pre phase()==ePhase::Finalize
   * \post phase()==ePhase::Ended
   */
  virtual void finalizeExchange() = 0;

  //! Mesh associated with this exchanger.
  virtual IPrimaryMesh* mesh() const = 0;

  //! Exchanger associated with the \a family. Throws an exception if not found.
  virtual IItemFamilyExchanger* findExchanger(IItemFamily* family) = 0;

  //! Phase of the exchange we are currently in.
  virtual ePhase phase() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
