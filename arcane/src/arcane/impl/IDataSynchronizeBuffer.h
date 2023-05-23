// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataSynchronizeBuffer.h                                    (C) 2000-2023 */
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
 * Cette instance contient des buffers d'envoie et de réception et peut être
 * utilisée quel que soit le type de donnée de la synchronisation.
 *
 * Chaque buffer est composé de \a nbRank() parties et chaque
 * partie est associée à un destinataire (sendBuffer() ou receiveBuffer()).
 *
 * Avant d'utiliser les buffers, il faut recopier les valeurs de la données
 * La méthode copySendAsync() permet de recopier les valeurs de la donnée dans le
 * buffer d'envoi et copyReceiveAsync() permet de recopier le buffer de réception
 * dans la donnée.
 *
 * \warning Ces méthodes copyReceiveAsync() et copySendAsync() peuvent être asynchrones.
 * Il est donc important d'appeler barrier() avant d'utiliser les données copiées
 * pour être sur que les transferts sont terminées.
 *
 * Si hasGlobalBuffer() est vrai alors les buffers de chaque partie sont issues
 * d'un buffer global et il est possible de le récupérer via globalSendBuffer()
 * pour l'envoi et globalReceiveBuffer() pour la réception. Il est aussi
 * possible dans ce de récupérer le déplacement de chaque sous-partie via
 * les méthodes sendDisplacement() ou receiveDisplacement().
 */
class ARCANE_IMPL_EXPORT IDataSynchronizeBuffer
{
 public:

  virtual ~IDataSynchronizeBuffer() = default;

 public:

  //! Nombre de rangs.
  virtual Int32 nbRank() const = 0;

  //! Indique si les buffers sont globaux.
  virtual bool hasGlobalBuffer() const = 0;

  //! Buffer d'envoi
  virtual MutableMemoryView globalSendBuffer() = 0;

  //! Buffer de réception
  virtual MutableMemoryView globalReceiveBuffer() = 0;

  //! Buffer d'envoi pour le rang \a index
  virtual MutableMemoryView sendBuffer(Int32 index) = 0;

  //! Buffer de réception pour le rang \a index
  virtual MutableMemoryView receiveBuffer(Int32 index) = 0;

  //! Déplacement (en octets) depuis le début de sendBuffer() pour le rang \a index
  virtual Int64 sendDisplacement(Int32 index) const = 0;

  //! Déplacement (en octets) depuis le début de receiveBuffer() pour le rang \a index
  virtual Int64 receiveDisplacement(Int32 index) const = 0;

  //! Recopie dans les données depuis le buffer de réception du rang \a index.
  virtual void copyReceiveAsync(Int32 index) = 0;

  //! Recopie toutes les données depuis le buffer de réception.
  virtual void copyAllReceive();

  //! Recopie dans le buffer d'envoi les données du rang \a index
  virtual void copySendAsync(Int32 index) = 0;

  //! Recopie dans le buffer d'envoi toute les données
  virtual void copyAllSend();

  //! Taille totale à envoyer en octet
  virtual Int64 totalSendSize() const = 0;

  //! Taille totale à recevoir en octet
  virtual Int64 totalReceiveSize() const = 0;

  //! Attend que les copies (copySendAsync() et copyReceiveAsync()) soient terminées
  virtual void barrier() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
