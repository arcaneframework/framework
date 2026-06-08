// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GatherMessageInfo.h                                         (C) 2000-2025 */
/*                                                                           */
/* Information for 'gather' messages.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_GATHERMESSAGEINFO_H
#define ARCCORE_MESSAGEPASSING_GATHERMESSAGEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/Array.h"
#include "arccore/message_passing/MessageRank.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Brief information for a 'gather' message.
 *
 * It is better to use the GatherMessageInfo class rather than this class.
 * This class allows using 'Gather', 'GatherVariable',
 * 'AllGather', and 'AllGatherVariable' messages generically.
 */
class ARCCORE_MESSAGEPASSING_EXPORT GatherMessageInfoBase
{
 public:

  //! Message mode
  enum class Mode
  {
    Gather,
    GatherVariable,
    GatherVariableNeedComputeInfo,
    Null
  };

 public:

  //! Message for everyone and blocking
  GatherMessageInfoBase() = default;

  //! Blocking message having destination \a rank
  explicit GatherMessageInfoBase(MessageRank dest_rank)
  : m_destination_rank(dest_rank)
  {}

  //! Message having destination \a dest_rank and blocking mode \a blocking_type
  GatherMessageInfoBase(MessageRank dest_rank, eBlockingType blocking_type)
  : m_destination_rank(dest_rank)
  , m_is_blocking(blocking_type == Blocking)
  {}

 public:

  void setBlocking(bool is_blocking)
  {
    m_is_blocking = is_blocking;
  }
  //! Indicates if the message is blocking.
  bool isBlocking() const { return m_is_blocking; }

  //! Rank of the message destination
  MessageRank destinationRank() const { return m_destination_rank; }

  //! Sets the rank of the message destination
  void setDestinationRank(MessageRank rank)
  {
    m_destination_rank = rank;
  }

  //! Message mode
  Mode mode() const { return m_mode; }

  //! Prints the message
  void print(std::ostream& o) const;

  friend std::ostream& operator<<(std::ostream& o, const GatherMessageInfoBase& pmessage)
  {
    pmessage.print(o);
    return o;
  }

 public:

  // Indicates if the message is valid (i.e.: it has been initialized
  // with a valid message)
  bool isValid() const
  {
    if (m_mode == Mode::Null)
      return false;
    return true;
  }

 protected:

  void _setType(Mode t)
  {
    m_mode = t;
  }

 private:

  MessageRank m_destination_rank;
  bool m_is_blocking = true;
  Mode m_mode = Mode::Null;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Brief information for a 'gather' message for data type \a DataType.
 *
 * One of the setGather() or setGatherVariable() methods must be called before
 * sending the corresponding message. The instances passed as arguments
 * to these two methods must remain alive until the message is complete.
 */
template <typename DataType>
class GatherMessageInfo
: public GatherMessageInfoBase
{
 public:

  using BaseClass = GatherMessageInfoBase;

 public:

  //! Message for everyone and blocking
  GatherMessageInfo() = default;

  //! Blocking message having destination \a rank
  explicit GatherMessageInfo(MessageRank dest_rank)
  : BaseClass(dest_rank)
  {}

  //! Message having destination \a dest_rank and blocking mode \a blocking_type
  GatherMessageInfo(MessageRank dest_rank, eBlockingType blocking_type)
  : BaseClass(dest_rank, blocking_type)
  {}

 public:

  /*!
   * \brief Brief message equivalent to MPI_Gather or MPI_Allgather.
   *
   * All ranks must provide a valid value for \a send_buf.
   * Only the destination rank must provide \a receive_buf. \a receive_buf
   * must be able to accommodate the size send_buf.size() * nb_rank.
   */
  void setGather(Span<const DataType> send_buf, Span<DataType> receive_buf)
  {
    _setType(Mode::Gather);
    m_receive_buf = receive_buf;
    m_send_buffer = send_buf;
  }

  /*!
   * \brief Brief message equivalent to MPI_Gatherv or MPI_Allgatherv.
   *
   * This prototype is used when it is unknown what each rank will send. If this
   * information is known, it is preferable to use
   * the setGatherVariable() method containing the displacements and message size
   * of each participant.
   *
   * Calling this method triggers a call to mpGather() to determine what
   * each participant must send. For this reason, it cannot be used
   * in blocking mode.
   *
   * Only the destination rank must provide \a receive_array. For others, it
   * is possible to use \a nullptr.
   */
  void setGatherVariable(Span<const DataType> send_buf, Array<DataType>* receive_array)
  {
    _setType(Mode::GatherVariableNeedComputeInfo);
    m_local_reception_buffer = receive_array;
    m_send_buffer = send_buf;
  }

  /*!
   * \brief Brief message equivalent to MPI_Gatherv or MPI_Allgatherv.
   *
   * All ranks must provide a valid value for \a send_buf,
   * \a receive_counts and \a receive_displacements.
   * Only the destination rank must provide \a receive_buf. \a receive_buf
   * must be able to accommodate the size send_buf.size() * nb_rank.
   */
  void setGatherVariable(Span<const DataType> send_buf, Span<DataType> receive_buf,
                         Span<const Int32> receive_counts, Span<const Int32> receive_displacements)
  {
    _setType(Mode::GatherVariable);
    m_receive_buf = receive_buf;
    m_send_buffer = send_buf;
    m_receive_displacements = receive_displacements;
    m_receive_counts = receive_counts;
  }

  /*!
   * \brief Receive buffer for the T_GatherVariableNeedComputeInfo type.
   *
   * May be null for ranks not involved in reception.
   */
  Array<DataType>* localReceptionBuffer() const { return m_local_reception_buffer; }

  //! Send buffer. It is used in all modes.
  Span<const DataType> sendBuffer() const { return m_send_buffer; }

  //! Receive buffer. Used in Gather and GatherVariable mode by ranks that receive
  Span<DataType> receiveBuffer() const { return m_receive_buf; }

  //! Displacement array. Used in GatherVariable mode.
  Span<const Int32> receiveDisplacement() { return m_receive_displacements; }

  //! Counts array. Used in GatherVariable mode.
  Span<const Int32> receiveCounts() const { return m_receive_counts; }

 private:

  Array<DataType>* m_local_reception_buffer = nullptr;
  Span<const DataType> m_send_buffer;
  Span<DataType> m_receive_buf;
  Span<const Int32> m_receive_displacements;
  Span<const Int32> m_receive_counts;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
