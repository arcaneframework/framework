// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelExchanger.h                                        (C) 2000-2012 */
/*                                                                           */
/* Echange d'informations entre processeurs.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IPARALLELEXCHANGER_H
#define ARCANE_IPARALLELEXCHANGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;
class IItemFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Echange d'informations entre processeurs.
 *
 * Cette classe permet d'envoyer et de recevoir des messages quelconques
 * d'un nombre quelconque d'autre processeurs.
 *
 * Le fonctionnement est le suivant.
 * - indiquer les autres PE avec lesquels on souhaite communiquer
 * en appelant addSender(), éventuellement plusieurs fois.
 * - appeller initializeCommunicationsMessages() pour
 * déterminer la liste des PE pour lesquels on doit recevoir des infos.
 * - pour chaque message d'envoie, sérialiser les informations qu'on souhaite
 * envoyer.
 * - effectuer les envoies et les réceptions en appelant processExchange()
 * - désérialiser les messages reçus.
 *
 * Il est possible de spécifier, avant appel à processExchange(), la manière dont
 * les messages seront envoyé via setExchangeMode(). Par défaut, le mécanisme
 * utilisé est celui des communications point à point.
 */
class ARCANE_CORE_EXPORT IParallelExchanger
{
 public:

  /*!
   * \brief Mode d'échange.
   *
   */
  enum eExchangeMode
  {
    //! Utilise les échanges point à point (send/recv)
    EM_Independant,
    //! Utilise les opération collectives (allToAll)
    EM_Collective,
    //! Choisi automatiquement entre point à point ou collective.
    EM_Auto
  };

 public:

  virtual ~IParallelExchanger(){}

 public:

  /*!
   * \brief Calcule les communications.
   *
   A partir de \a m_send_ranks donné par chaque processeur,
   détermine la liste des processeurs à qui on doit envoyer un message.

   Afin de connaître les processeurs desquels on attend des informations,
   il est nécessaire de faire une communication (allGatherVariable()). Si on connait
   à priori ces processeurs, il faut utiliser une des versions surchargée de cette
   méthode.

   \retval true s'il n'y a rien à échanger
   \retval false sinon.
  */
  virtual bool initializeCommunicationsMessages() =0;

  /*! \brief Calcule les communications.
   *
   * Suppose que la liste des processeurs dont on veut les informations est dans
   * \a recv_ranks.
   */
  virtual void initializeCommunicationsMessages(Int32ConstArrayView recv_ranks) =0;

  //! Effectue l'échange
  virtual void processExchange() =0;

 public:
 
  virtual IParallelMng* parallelMng() const =0;

  //! Nombre de processeurs auquel on envoie
  virtual Integer nbSender() const =0;
  //! Liste des rangs des processeurs auquel on envoie
  virtual Int32ConstArrayView senderRanks() const =0;
  //! Ajoute un processeur à envoyer
  virtual void addSender(Int32 rank) =0;
  //! Message destiné au \a ième processeur
  virtual ISerializeMessage* messageToSend(Integer i) =0;

  //! Nombre de processeurs dont on va réceptionner les messages
  virtual Integer nbReceiver() const =0;
  //! Liste des rangs des processeurs dont on va réceptionner les messages
  virtual Int32ConstArrayView receiverRanks() =0;
  //! Message reçu du \a ième processeur
  virtual ISerializeMessage* messageToReceive(Integer i) =0;

  /*!
   * \brief Indique le mode d'échange.
   */
  virtual void setExchangeMode(eExchangeMode mode) =0;

  //! Mode d'échange spécifié
  virtual eExchangeMode exchangeMode() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

