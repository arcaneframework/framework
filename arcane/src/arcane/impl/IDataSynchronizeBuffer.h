// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataSynchronizeBuffer.h                                    (C) 2000-2023 */
/*                                                                           */
/* Interface of a generic buffer for data synchronization.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_IDATASYNCHRONIZEBUFFER_H
#define ARCANE_IMPL_IDATASYNCHRONIZEBUFFER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Generic buffer for data synchronization.
 *
 * This instance contains send and receive buffers and can be used regardless
 * of the data type of the synchronization.
 *
 * Each buffer is composed of \a nbRank() parts and each part is associated
 * with a recipient (sendBuffer() or receiveBuffer()).
 *
 * Before using the buffers, the data values must be copied.
 * The copySendAsync() method allows copying the data values into the
 * send buffer and copyReceiveAsync() allows copying the receive buffer
 * into the data.
 *
 * \warning These copyReceiveAsync() and copySendAsync() methods may be asynchronous.
 * It is therefore important to call barrier() before using the copied
 * data to ensure that the transfers are complete.
 *
 * If hasGlobalBuffer() is true, then the buffers of each part come from
 * a global buffer and it is possible to retrieve it via globalSendBuffer()
 * for sending and globalReceiveBuffer() for receiving. It is also
 * possible in this case to retrieve the displacement of each sub-part via
 * the methods sendDisplacement() or receiveDisplacement().
 */
class ARCANE_IMPL_EXPORT IDataSynchronizeBuffer
{
 public:

  virtual ~IDataSynchronizeBuffer() = default;

 public:

  //! Number of ranks.
  virtual Int32 nbRank() const = 0;

  //! Target rank of the \a \a index-th rank
  virtual Int32 targetRank(Int32 index) const = 0;

  //! Indicates if the buffers are global.
  virtual bool hasGlobalBuffer() const = 0;

  //! Send buffer
  virtual MutableMemoryView globalSendBuffer() = 0;

  //! Receive buffer
  virtual MutableMemoryView globalReceiveBuffer() = 0;

  //! Send buffer for the \a index-th rank
  virtual MutableMemoryView sendBuffer(Int32 index) = 0;

  //! Receive buffer for the \a index-th rank
  virtual MutableMemoryView receiveBuffer(Int32 index) = 0;

  /*!
   * \brief Displacement (in bytes) from the start of sendBuffer()
   * for the \a index-th rank.
   *
   * This value is only meaningful if hasGlobalBuffer() is true.
   */
  virtual Int64 sendDisplacement(Int32 index) const = 0;

  /*!
   * \brief Displacement (in bytes) from the start of receiveBuffer()
   * for the \a index-th rank.
   *
   * This value is only meaningful if hasGlobalBuffer() is true.
   */
  virtual Int64 receiveDisplacement(Int32 index) const = 0;

  //! Copies into the data from the receive buffer of the \a index-th rank.
  virtual void copyReceiveAsync(Int32 index) = 0;

  /*!
   * \brief Copies all data from the receive buffer.
   *
   * This call is equivalent to:
   * \code
   * for (Int32 i = 0; i < nb_rank; ++i)
   *   copySendAsync(i);
   * barrier()
   * \endcode
   */
  virtual void copyAllReceive();

  /*!
   * \brief Copies the data of the \a index-th rank into the send buffer.
   *
   * This call is equivalent to:
   * \code
   * for (Int32 i = 0; i < nb_rank; ++i)
   *   copyReceiveAsync(i);
   * barrier()
   * \endcode
   */
  virtual void copySendAsync(Int32 index) = 0;

  /*!
   * \brief Copies all data into the send buffer.
   */
  virtual void copyAllSend();

  //! Total size to send in bytes
  virtual Int64 totalSendSize() const = 0;

  //! Total size to receive in bytes
  virtual Int64 totalReceiveSize() const = 0;

  //! Waits until the copies (copySendAsync() and copyReceiveAsync()) are finished
  virtual void barrier() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
