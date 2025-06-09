// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilySerializeStep.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface d'une étape de la sérialisation des familles d'entités.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMFAMILYSERIALIZESTEP_H
#define ARCANE_CORE_IITEMFAMILYSERIALIZESTEP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une étape de la sérialisation des familles d'entités.
 *
 * Cette interface est utilisée par IItemFamilyExchanger pour sérialiser
 * et désérialiser des informations. La sérialisation se fait par échange
 * de messages et il y a un message par rang avec lequel on communique.
 *
 * Le pseudo-code d'appel est le suivant:
 \code
 * IItemFamilyExchanger* exchanger = ...;
 * IItemFamilySerializeStep* step = ...;
 * exchanger->computeExchangeInfos();
 * step->initialize();
 * // Some exchanger action
 * ...
 * step->notifyAction(AC_BeginPrepareSend);
 * // Set serialize mode to ISerializer::ModeReserve
 * for( Integer i=0; i<nb_message; ++i )
 *   step->serialize(i);
 * // Set serialize mode to ISerializer::ModePut
 * for( Integer i=0; i<nb_message; ++i )
 *   step->serialize(i);
 * step->notifyAction(AC_EndPrepareSend);
 * exchanger->processExchange();
 * step->notifyAction(AC_BeginReceive);
 * // Set serialize mode to ISerializer::ModeGet
 * for( Integer i=0; i<nb_message; ++i )
 *   step->serialize(i);
 * step->notifyAction(AC_EndReceive);
 * step->finalize();
 \endcode
 *
 * La méthode serialize() est appelée pour chaque rang avec lequel on
 * communique.
 *
 * L'étape est appelé lors de la phase de sérialisation spécifiée par phase()
 * comme spécifié dans la documentation de IItemFamilyExchanger.
 *
 * Pour la phase spécifiée, l'ordre d'appel est le suivant:
 \code
 * IItemFamilySerializeStep* step = ...;
 * ISerializer* sbuf = ...;
 * sbuf->setMode(ISerializer::ModeReserve)
 * step->beginSerialize(sbuf->mode())
 \endcode
 */
class ARCANE_CORE_EXPORT IItemFamilySerializeStep
{
 public:

  //! Phase de la sérialisation
  enum ePhase
  {
    PH_Item,
    PH_Group,
    PH_Variable
  };
  //! Action en cours de la sérialisation
  enum class eAction
  {
    //! Début de la préparation de l'envoie.
    AC_BeginPrepareSend,
    //! Fin de la préparation de l'envoie.
    AC_EndPrepareSend,
    //! Début de la réception des données.
    AC_BeginReceive,
    //! Fin de la réception des données.
    AC_EndReceive,
  };

 public:

  class NotifyActionArgs
  {
   public:

    NotifyActionArgs(eAction aaction, Integer nb_message)
    : m_action(aaction)
    , m_nb_message(nb_message)
    {}

   public:

    eAction action() const { return m_action; }
    //! Nombre de messages de sérialisation
    Integer nbMessage() const { return m_nb_message; }

   private:

    eAction m_action;
    Integer m_nb_message;
  };

 public:

  virtual ~IItemFamilySerializeStep() = default;

 public:

  //! Initialise l'instance avant le début des échanges.
  virtual void initialize() = 0;

  //! Notifie l'instance qu'on entre dans une certaine phase de l'échange.
  virtual void notifyAction(const NotifyActionArgs& args) = 0;

  /*!
   * \brief Sérialise dans/depuis \a buf.
   *
   * \a args.rank() contient le rang du sous-domaine avec lequel on
   * communique. \a args.messageIndex() l'index numéro du message et
   * \a args.nbMessageIndex() le nombre de message qui seront envoyés.
   *
   * En sérialisation, il s'agit des indices locaux des entités envoyées au
   * rang \a rank(). En désérialisation, il s'agit des indices locaux
   * recues par le rang \a rank().
   */
  virtual void serialize(const ItemFamilySerializeArgs& args) = 0;

  //! Effectue les traitements de fin d'échange.
  virtual void finalize() = 0;

  //! Phase de la sérialisation où cette instance intervient.
  virtual ePhase phase() const = 0;

  //! Famille associée
  virtual IItemFamily* family() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fabrique pour créer une étape de la sérialisation des
 * familles d'entités.
 */
class ARCANE_CORE_EXPORT IItemFamilySerializeStepFactory
{
 public:

  virtual ~IItemFamilySerializeStepFactory() = default;

 public:

  /*!
   * \brief Créé une étape pour la famille \a family.
   *
   * Peut retourner nullptr auquel cas aucune étape n'est ajoutée pour cette
   * fabrique.
   */
  virtual IItemFamilySerializeStep* createStep(IItemFamily* family) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

