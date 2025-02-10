// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GatherMessageInfo.h                                         (C) 2000-2025 */
/*                                                                           */
/* Informations pour les messages 'gather'.                                  */
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
/*!
 * \brief Informations pour un message 'gather'.
 *
 * Il faut plutôt utiliser la classe GatherMessageInfo plutôt que cette classe.
 * Cette classe permet d'utiliser les messages de type 'Gather', 'GatherVariable',
 * 'AllGather' et 'AllGatherVariable' de manière générique.
 */
class ARCCORE_MESSAGEPASSING_EXPORT GatherMessageInfoBase
{
 public:

  //! Mode du message
  enum class Mode
  {
    Gather,
    GatherVariable,
    GatherVariableNeedComputeInfo,
    Null
  };

 public:

  //! Message pout tout le monde et bloquant
  GatherMessageInfoBase() = default;

  //! Message bloquant ayant pour destination \a rank
  explicit GatherMessageInfoBase(MessageRank dest_rank)
  : m_destination_rank(dest_rank)
  {}

  //! Message ayant pour destination \a dest_rank et mode bloquant \a blocking_type
  GatherMessageInfoBase(MessageRank dest_rank, eBlockingType blocking_type)
  : m_destination_rank(dest_rank)
  , m_is_blocking(blocking_type == Blocking)
  {}

 public:

  void setBlocking(bool is_blocking)
  {
    m_is_blocking = is_blocking;
  }
  //! Indique si le message est bloquant.
  bool isBlocking() const { return m_is_blocking; }

  //! Rang de la destination du message
  MessageRank destinationRank() const { return m_destination_rank; }

  //! Positionne le rang de la destination du message
  void setDestinationRank(MessageRank rank)
  {
    m_destination_rank = rank;
  }

  //! Mode du message
  Mode mode() const { return m_mode; }

  //! Affiche le message
  void print(std::ostream& o) const;

  friend std::ostream& operator<<(std::ostream& o, const GatherMessageInfoBase& pmessage)
  {
    pmessage.print(o);
    return o;
  }

 public:

  // Indique si le message est valide (i.e: il a été initialisé avec un message valide)
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
 * \brief Informations pour un message 'gather' pour le type de données \a DataType.
 *
 * Il faut appeler une des méthodes setGather() ou setGatherVariable() avant
 * de pouvoir envoyer le message correspondant. Les instances passées en argument
 * ces deux méthodes doivent rester vivantes tant que le message n'est pas terminé.
 */
template <typename DataType>
class GatherMessageInfo
: public GatherMessageInfoBase
{
 public:

  using BaseClass = GatherMessageInfoBase;

 public:

  //! Message pout tout le monde et bloquant
  GatherMessageInfo() = default;

  //! Message bloquant ayant pour destination \a rank
  explicit GatherMessageInfo(MessageRank dest_rank)
  : BaseClass(dest_rank)
  {}

  //! Message ayant pour destination \a dest_rank et mode bloquant \a blocking_type
  GatherMessageInfo(MessageRank dest_rank, eBlockingType blocking_type)
  : BaseClass(dest_rank, blocking_type)
  {}

 public:

  /*!
   * \brief Message équivalent à MPI_Gather ou MPI_Allgather.
   *
   * Tous les rangs doivent positionner une valeur valide pour \a send_buf.
   * Seul le rang destinataire doit positionner \a receive_buf. \a receive_buf
   * doit pouvoir pour taille send_buf.size() * nb_rank.
   */
  void setGather(Span<const DataType> send_buf, Span<DataType> receive_buf)
  {
    _setType(Mode::Gather);
    m_receive_buf = receive_buf;
    m_send_buffer = send_buf;
  }

  /*!
   * \brief Message équivalent à MPI_Gatherv ou MPI_Allgatherv.
   *
   * Ce prototype est utilisé lorsqu'on ne connait pas ce que va envoyer
   * chaque rang. Si on connait cette information il est préférable d'utiliser
   * la méthode setGatherVariable() contenant les déplacements et la taille de message
   * de chaque participant.
   *
   * L'appel à cette méthode provoque un appel à mpGather() pour déterminer ce que
   * chaque participant doit envoyer. Pour cette raison elle ne peut pas être utilisée
   * en mode bloquant.
   *
   * Seul le rang destinataire doit positionner \a receive_array. Pour les autres il
   * est possible d'utiliser \a nullptr.
   */
  void setGatherVariable(Span<const DataType> send_buf, Array<DataType>* receive_array)
  {
    _setType(Mode::GatherVariableNeedComputeInfo);
    m_local_reception_buffer = receive_array;
    m_send_buffer = send_buf;
  }

  /*!
   * \brief Message équivalent à MPI_Gatherv ou MPI_Allgatherv.
   *
   * Tous les rangs doivent positionner une valeur valide pour \a send_buf,
   * \a receive_counts et \a receive_displacements.
   * Seul le rang destinataire doit positionner \a receive_buf. \a receive_buf
   * doit pouvoir pour taille send_buf.size() * nb_rank.
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
   * \brief Buffer de réception pour le type T_GatherVariableNeedComputeInfo.
   *
   * Peut-être nul pour les rangs qui ne sont pas concernés par la réception.
   */
  Array<DataType>* localReceptionBuffer() const { return m_local_reception_buffer; }

  //! Buffer d'envoi. Il est utilisé dans tous les modes.
  Span<const DataType> sendBuffer() const { return m_send_buffer; }

  //! Buffer de réception. Utilisé en mode Gather et GatherVariable par les rangs qui recoivent
  Span<DataType> receiveBuffer() const { return m_receive_buf; }

  //! Tableau des déplacements. Utilisé en mode GatherVariable.
  Span<const Int32> receiveDisplacement() { return m_receive_displacements; }

  //! Tableau des tailles. Utilisé en mode GatherVariable.
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

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
