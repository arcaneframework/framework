// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelMngUtilsFactory.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface of a factory for the utility functions of IParallelMng.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IPARALLELMNGUTILSFACTORY_H
#define ARCANE_CORE_INTERNAL_IPARALLELMNGUTILSFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"

#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a factory for the utility functions of IParallelMng.
 */
class ARCANE_CORE_EXPORT IParallelMngUtilsFactory
{
 public:

  virtual ~IParallelMngUtilsFactory() = default;

 public:

  /*!
   * \brief Returns an operation to retrieve the values of a variable
   * on the entities of another subdomain.
   */
  virtual Ref<IGetVariablesValuesParallelOperation>
  createGetVariablesValuesOperation(IParallelMng* pm) = 0;

  //! Returns an operation to transfer values between ranks.
  virtual Ref<ITransferValuesParallelOperation>
  createTransferValuesOperation(IParallelMng* pm) = 0;

  //! Returns an interface to transfer messages between ranks
  virtual Ref<IParallelExchanger>
  createExchanger(IParallelMng* pm) = 0;

  /*!
   * \brief Returns an interface to synchronize
   * variables on the group of the family \a family
   */
  virtual Ref<IVariableSynchronizer>
  createSynchronizer(IParallelMng* pm, IItemFamily* family) = 0;

  /*!
   * \brief Returns an interface to synchronize
   * variables on the group \a group.
   */
  virtual Ref<IVariableSynchronizer>
  createSynchronizer(IParallelMng* pm, const ItemGroup& group) = 0;

  /*!
   * \brief Creates an instance containing information about the rank topology of this manager.
   *
   * This operation is collective.
   */
  virtual Ref<IParallelTopology>
  createTopology(IParallelMng* pm) = 0;

  /*!
   * \brief Creates a non-blocking serialization message for sending to rank \a rank.
   *
   * The message is processed only when IParallelMng::processMessages() is called.
   */
  virtual Ref<ISerializeMessage>
  createSendSerializeMessage(IParallelMng* pm, Int32 rank) = 0;

  /*!
   * \brief Creates a non-blocking serialization message for receiving from rank \a rank.
   *
   * The message is processed only when IParallelMng::processMessages() is called.
   */
  virtual Ref<ISerializeMessage>
  createReceiveSerializeMessage(IParallelMng* pm, Int32 rank) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
