// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilySerializeStep.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface for a step in the serialization of entity families.             */
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
 * \brief Interface for a step in the serialization of entity families.
 *
 * This interface is used by IItemFamilyExchanger to serialize
 * and deserialize information. Serialization is done by message exchange
 * and there is one message per rank with which we communicate.
 *
 * The call pseudo-code is as follows:
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
 * The serialize() method is called for each rank we communicate with.
 *
 * The step is called during the serialization phase specified by phase()
 * as specified in the IItemFamilyExchanger documentation.
 *
 * For the specified phase, the call order is as follows:
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

  //! Serialization phase
  enum ePhase
  {
    PH_Item,
    PH_Group,
    PH_Variable
  };
  //! Action during serialization
  enum class eAction
  {
    //! Start of send preparation.
    AC_BeginPrepareSend,
    //! End of send preparation.
    AC_EndPrepareSend,
    //! Start of data reception.
    AC_BeginReceive,
    //! End of data reception.
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
    //! Number of serialization messages
    Integer nbMessage() const { return m_nb_message; }

   private:

    eAction m_action;
    Integer m_nb_message;
  };

 public:

  virtual ~IItemFamilySerializeStep() = default;

 public:

  //! Initializes the instance before the start of exchanges.
  virtual void initialize() = 0;

  //! Notifies the instance that we are entering a certain phase of the exchange.
  virtual void notifyAction(const NotifyActionArgs& args) = 0;

  /*!
   * \brief Serializes into/from \a buf.
   *
   * \a args.rank() contains the rank of the subdomain with which we
   * communicate. \a args.messageIndex() is the message number index and
   * \a args.nbMessageIndex() is the number of messages that will be sent.
   *
   * During serialization, these are the local indices of the entities sent to
   * rank \a rank(). During deserialization, these are the local indices
   * received by rank \a rank().
   */
  virtual void serialize(const ItemFamilySerializeArgs& args) = 0;

  //! Performs end-of-exchange processing.
  virtual void finalize() = 0;

  //! Serialization phase where this instance is involved.
  virtual ePhase phase() const = 0;

  //! Associated family
  virtual IItemFamily* family() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Factory for creating a step in the serialization of
 * entity families.
 */
class ARCANE_CORE_EXPORT IItemFamilySerializeStepFactory
{
 public:

  virtual ~IItemFamilySerializeStepFactory() = default;

 public:

  /*!
   * \brief Creates a step for the family \a family.
   *
   * May return nullptr in which case no step is added for this
   * factory.
   */
  virtual IItemFamilySerializeStep* createStep(IItemFamily* family) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
