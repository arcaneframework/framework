// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelMngUtils.h                                          (C) 2000-2026 */
/*                                                                           */
/* Utility functions associated with 'IParallelMng'.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PARALLELMNGUTILS_H
#define ARCANE_CORE_PARALLELMNGUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Utility functions associated with IParallelMng.
 */
namespace Arcane::ParallelMngUtils
{

/*!
 * \brief Returns an operation to retrieve the values of a variable
 * on the entities of another subdomain.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IGetVariablesValuesParallelOperation>
createGetVariablesValuesOperationRef(IParallelMng* pm);

//! Returns an operation to transfer values between ranks.
extern "C++" ARCANE_CORE_EXPORT Ref<ITransferValuesParallelOperation>
createTransferValuesOperationRef(IParallelMng* pm);

//! Returns an interface to transfer messages between ranks
extern "C++" ARCANE_CORE_EXPORT Ref<IParallelExchanger>
createExchangerRef(IParallelMng* pm);

/*!
 * \brief Returns an interface to synchronize
 * variables on the group of the family \a family
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IVariableSynchronizer>
createSynchronizerRef(IParallelMng* pm, IItemFamily* family);

/*!
 * \brief Returns an interface to synchronize
 * variables on the group \a group.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IVariableSynchronizer>
createSynchronizerRef(IParallelMng* pm, const ItemGroup& group);

/*!
 * \brief Creates an instance containing information about the rank topology of this manager.
 *
 * This operation is collective.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IParallelTopology>
createTopologyRef(IParallelMng* pm);

/*!
 * \brief Creates a new parallelism manager for a subset
 * of ranks.
 *
 * This operation is collective and is equivalent to MPI_Comm_split.
 *
 * Ranks whose \a color has the same value will be in the same communicator.
 * \a key allows ordering the ranks in the created sub-communicator. If it equals
 * pm->commRank(), then the ranks in the sub-communicator will have the same order
 * as in \a pm.
 *
 * * If \a color is negative, then the current rank will not be associated with any
 * communicator and the returned value will be null.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IParallelMng>
createSubParallelMngRef(IParallelMng* pm, Int32 color, Int32 key);

/*!
 * \brief Creates a non-blocking serialization message for sending to rank \a rank.
 *
 * The message is processed only when IParallelMng::processMessages() is called.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<ISerializeMessage>
createSendSerializeMessageRef(IParallelMng* pm, Int32 rank);

/*!
 * \brief Creates a non-blocking serialization message for receiving from rank \a rank.
 *
 * The message is processed only when IParallelMng::processMessages() is called.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<ISerializeMessage>
createReceiveSerializeMessageRef(IParallelMng* pm, Int32 rank);

/*!
 * \brief Function to determine if shared memory window mode
 * is supported and if its use is possible.
 *
 * Collective call.
 *
 * This function can be useful for using the classes:
 * - MachineShMemWin(Base)
 * - ContigMachineShMemWin(Base)
 * - MachineShMemWinVariable...
 *
 * \return true if the classes above can be used.
 */
extern "C++" ARCANE_CORE_EXPORT bool
isMachineShMemWinAvailable(IParallelMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::ParallelMngUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
