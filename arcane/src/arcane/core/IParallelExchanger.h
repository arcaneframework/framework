// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelExchanger.h                                        (C) 2000-2025 */
/*                                                                           */
/* Échange d'informations entre processeurs.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARALLELEXCHANGER_H
#define ARCANE_CORE_IPARALLELEXCHANGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"
#include "arcane/core/ParallelExchangerOptions.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Échange d'informations entre processeurs.
 *
 * Cette classe permet d'envoyer et de recevoir des messages quelconques
 * d'un nombre quelconque d'autre processeurs.
 *
 * Le fonctionnement est le suivant.
 *
 * 1. indiquer les autres PE avec lesquels on souhaite communiquer en appelant
 *    addSender(), éventuellement plusieurs fois.
 * 2. appeler initializeCommunicationsMessages() pour déterminer la liste des
 *    PE pour lesquels on doit recevoir des infos. Il existe deux surcharges
 *    pour cette méthode suivant si on connait ou non le nombre de rangs pour
 *    lesquels on doit recevoir des informations.
 * 3. pour chaque message d'envoi, sérialiser les informations qu'on souhaite
 *    envoyer.
 * 4. effectuer les envoies et les réceptions en appelant processExchange()
 * 5. récupérer les messages recus (via messageToReceive()) et désérialiser
 *    leurs informations.
 *
 * Il est possible de spécifier, avant appel à processExchange(), la manière dont
 * les messages seront envoyés via setExchangeMode(). Par défaut, le mécanisme
 * utilisé est celui des communications point à point (EM_Independant) mais il
 * est possible d'utiliser un mode collectif (EM_Collective) qui utilise
 * des messages de type 'all to all'.
 */
class ARCANE_CORE_EXPORT IParallelExchanger
{
 public:

  enum eExchangeMode
  {
    //! Utilise les échanges point à point (send/recv)
    EM_Independant = ParallelExchangerOptions::EM_Independant,
    //! Utilise les opération collectives (allToAll)
    EM_Collective = ParallelExchangerOptions::EM_Collective,
    //! Choisi automatiquement entre point à point ou collective.
    EM_Auto = ParallelExchangerOptions::EM_Auto
  };

 public:

  virtual ~IParallelExchanger() = default;

 public:

  /*!
   * \brief Calcule les communications.
   *
   * A partir de \a m_send_ranks donné par chaque processeur,
   * détermine la liste des processeurs à qui on doit envoyer un message.
   *
   * Afin de connaître les processeurs desquels on attend des informations,
   * il est nécessaire de faire une communication (allGatherVariable()). Si on connait
   * à priori ces processeurs, il faut utiliser une des versions surchargée de cette
   * méthode.
   *
   * \retval true s'il n'y a rien à échanger
   * \retval false sinon.
   */
  virtual bool initializeCommunicationsMessages() = 0;

  /*! \brief Calcule les communications.
   *
   * Suppose que la liste des processeurs dont on veut les informations est dans
   * \a recv_ranks.
   */
  virtual void initializeCommunicationsMessages(Int32ConstArrayView recv_ranks) = 0;

  //! Effectue l'échange avec les options par défaut de ParallelExchangerOptions.
  virtual void processExchange() = 0;

  //! Effectue l'échange avec les options \a options
  virtual void processExchange(const ParallelExchangerOptions& options) = 0;

 public:

  virtual IParallelMng* parallelMng() const = 0;

  //! Nombre de processeurs auquel on envoie
  virtual Integer nbSender() const = 0;
  //! Liste des rangs des processeurs auquel on envoie
  virtual Int32ConstArrayView senderRanks() const = 0;
  //! Ajoute un processeur à envoyer
  virtual void addSender(Int32 rank) = 0;
  //! Message destiné au \a ième processeur
  virtual ISerializeMessage* messageToSend(Integer i) = 0;

  //! Nombre de processeurs dont on va réceptionner les messages
  virtual Integer nbReceiver() const = 0;
  //! Liste des rangs des processeurs dont on va réceptionner les messages
  virtual Int32ConstArrayView receiverRanks() = 0;
  //! Message reçu du \a ième processeur
  virtual ISerializeMessage* messageToReceive(Integer i) = 0;

  //! Positionne le mode d'échange.
  [[deprecated("Y2021: Use ParallelExchangerOptions::setExchangeMode()")]]
  virtual void setExchangeMode(eExchangeMode mode) = 0;
  //! Mode d'échange spécifié
  [[deprecated("Y2021: Use ParallelExchangerOptions::exchangeMode()")]]
  virtual eExchangeMode exchangeMode() const = 0;

  //! Positionne le niveau de verbosité
  virtual void setVerbosityLevel(Int32 v) = 0;
  //! Niveau de verbosité
  virtual Int32 verbosityLevel() const = 0;

  //! Positionne le nom de l'instance. Ce nom est utilisé lors des impressions
  virtual void setName(const String& name) = 0;
  //! Nom de l'instance
  virtual String name() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

