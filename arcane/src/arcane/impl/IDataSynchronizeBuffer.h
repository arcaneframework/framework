// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataSynchronizeBuffer.h                                    (C) 2000-2022 */
/*                                                                           */
/* Interface d'un buffer générique pour la synchronisation de donnéess.      */
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
 * \brief Buffer générique pour la synchronisation de données.
 *
 * Cette instance peut être utilisée quel que soit le type de donnée de la synchronisation.
 * On utilise un buffer pour l'envoi (globalSendBuffer()) et un pour la réception
 * (globalReceiveBuffer()). Chaque buffer est composé de \a nbRank() parties, chaque
 * partie étant associée à un destinataire (sendBuffer() ou receiveBuffer()).
 *
 */
class ARCANE_IMPL_EXPORT IDataSynchronizeBuffer
{
 public:

  virtual ~IDataSynchronizeBuffer() = default;

 public:

  //! Nombre de rangs.
  virtual Int32 nbRank() const = 0;
  //! Buffer d'envoi
  virtual Span<std::byte> globalSendBuffer() = 0;
  //! Buffer de réception
  virtual Span<std::byte> globalReceiveBuffer() = 0;
  //! Buffer d'envoi pour le rang \a index
  virtual Span<std::byte> sendBuffer(Int32 index) = 0;
  //! Buffer de réception pour le rang \a index
  virtual Span<std::byte> receiveBuffer(Int32 index) = 0;
  //! Déplacement depuis le début de sendBuffer() pour le rang \a index
  virtual Int64 sendDisplacement(Int32 index) const = 0;
  //! Déplacement depuis le début de receiveBuffer() pour le rang \a index
  virtual Int64 receiveDisplacement(Int32 index) const = 0;
  //! Recopie dans les données depuis le buffer de réception.
  virtual void copyReceive(Int32 index) = 0;
  //! Recopie dans le buffer d'envoi les données
  virtual void copySend(Int32 index) = 0;
  //! Taille totale à envoyer en octet
  virtual Int64 totalSendSize() const = 0;
  //! Taille totale à recevoir en octet
  virtual Int64 totalReceiveSize() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
