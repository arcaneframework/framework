// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITransferValuesParallelOperation.h                          (C) 2000-2025 */
/*                                                                           */
/* Value transfer across different processors.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITRANSFERVALUESPARALLELOPERATION_H
#define ARCANE_CORE_ITRANSFERVALUESPARALLELOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Sends values across different processors.
 *
 * This operation allows values to be communicated with other
 * processors. The array \a ranks indicates for each element the rank
 * of the processor it is intended for. It is then possible to specify
 * arrays containing the values to send and to receive. The
 * send arrays must have the same number of elements as \a ranks
 *
 * An instance is used only once. Once the transfer is complete,
 * it can be destroyed.
 *
 * For example, for a case with 3 processors:
 * \code
 * // Processor of rank 0:
 * Int32UniqueArray ranks;
 * ranks.add(2); // Sends to rank 2
 * ranks.add(1); // Sends to rank 1
 * ranks.add(1); // Sends to rank 1
 * Int32UniqueArray values_1;
 * values_1.add(5); // Sends 5 to rank 2 (ranks[0])
 * values_1.add(7); // Sends 7 to rank 1 (ranks[1])
 * values_1.add(6); // Sends 6 to rank 1 (ranks[2])
 * Int64UniqueArray values_2;
 * values_2.add(-5); // Sends -5 to rank 2 (ranks[0])
 * values_2.add(-7); // Sends -7 to rank 1 (ranks[1])
 * values_2.add(-6); // Sends -6 to rank 1 (ranks[2])

 * // Processor of rank 1:
 * Int32UniqueArray ranks;
 * ranks.add(0); // Sends to rank 0
 * ranks.add(2); // Sends to rank 2
 * Int32UniqueArray values_1;
 * values_1.add(1); // Sends 1 to rank 0 (ranks[0])
 * values_1.add(3); // Sends 3 to rank 2 (ranks[1])
 * Int64UniqueArray values_2;
 * values_2.add(23); // Sends 23 to rank 0 (ranks[0])
 * values_2.add(24); // Sends 24 to rank 2 (ranks[1])

 * // Processor of rank 2:
 * Int32UniqueArray ranks;
 * ranks.add(0); // Sends to rank 0
 * ranks.add(0); // Sends to rank 0
 * Int32UniqueArray values_1;
 * values_1.add(0); // Sends 1 to rank 0 (ranks[0])
 * values_1.add(4); // Sends 3 to rank 0 (ranks[1])
 * Int64UniqueArray values_2;
 * values_2.add(-1); // Sends -1 to rank 0 (ranks[0])
 * values_2.add(4); // Sends 4 to rank 0 (ranks[1])

 * \endcode
 *
 * To perform the transfer
 *
 * \code
 * Int32UniqueArray recv_values_1;
 * Int64UniqueArray recv_values_2;
 * op->setTransferRanks(ranks);
 * op->addArray(values_1,recv_values_1);
 * op->addArray(values_2,recv_values_2);
 * op->transferValues();
 * \endcode
 *
 * After sending, processor of rank 0 will have the following values:
 * \code
 * recv_values_1[0] == 1; // sent by rank 1
 * recv_values_1[1] == 0; // sent by rank 2
 * recv_values_1[2] == 4; // sent by rank 2
 * recv_values_2[0] == 23; // sent by rank 1
 * recv_values_2[1] == -1; // sent by rank 2
 * recv_values_2[2] == 4; // sent by rank 2
 * \endcode
 *
 * Note that the order of elements is undetermined
 */
class ARCANE_CORE_EXPORT ITransferValuesParallelOperation
{
 public:

  //! Destructor
  virtual ~ITransferValuesParallelOperation() = default;

 public:

  //! Associated parallelism manager
  virtual IParallelMng* parallelMng() = 0;

 public:

  //! Positions the array indicating who to send the values to.
  virtual void setTransferRanks(Int32ConstArrayView ranks) = 0;
  //! Adds an array of \c Int32
  virtual void addArray(Int32ConstArrayView send_values, SharedArray<Int32> recv_value) = 0;
  //! Adds an array of \c Int64
  virtual void addArray(Int64ConstArrayView send_values, SharedArray<Int64> recv_values) = 0;
  //! Adds an array of \c Int64
  virtual void addArray(RealConstArrayView send_values, SharedArray<Real> recv_values) = 0;
  /*!
   * \brief Sends and receives values.
   *
   * This call is collective and blocking.
   */
  virtual void transferValues() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
