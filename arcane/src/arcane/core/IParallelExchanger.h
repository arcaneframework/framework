// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelExchanger.h                                        (C) 2000-2025 */
/*                                                                           */
/* Information exchange between processors.                                  */
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
 * \brief Information exchange between processors.
 *
 * This class allows sending and receiving arbitrary messages
 * from any number of other processors.
 *
 * The operation is as follows.
 *
 * 1. indicate the other PEs you wish to communicate with by calling
 *    addSender(), possibly multiple times.
 * 2. call initializeCommunicationsMessages() to determine the list of
 *    PEs from which we must receive information. There are two overloads
 *    for this method depending on whether we know the number of ranks for
 *    which we must receive information.
 * 3. for each outgoing message, serialize the information you wish
 *    to send.
 * 4. perform the sends and receives by calling processExchange()
 * 5. retrieve the received messages (via messageToReceive()) and deserialize
 *    their information.
 *
 * It is possible to specify, before calling processExchange(), how
 * the messages will be sent via setExchangeMode(). By default, the mechanism
 * used is that of point-to-point communications (EM_Independant) but it
 * is possible to use a collective mode (EM_Collective) which uses
 * 'all to all' type messages.
 */
class ARCANE_CORE_EXPORT IParallelExchanger
{
 public:

  enum eExchangeMode
  {
    //! Uses point-to-point exchanges (send/recv)
    EM_Independant = ParallelExchangerOptions::EM_Independant,
    //! Uses collective operations (allToAll)
    EM_Collective = ParallelExchangerOptions::EM_Collective,
    //! Automatically chooses between point-to-point or collective.
    EM_Auto = ParallelExchangerOptions::EM_Auto
  };

 public:

  virtual ~IParallelExchanger() = default;

 public:

  /*!
   * \brief Calculates communications.
   *
   * Based on \a m_send_ranks provided by each processor,
   * determines the list of processors to which a message must be sent.
   *
   * To know the processors from which information is expected,
   * it is necessary to perform a communication (allGatherVariable()). If we know
   * these processors beforehand, we must use one of the overloaded versions of this
   * method.
   *
   * \retval true if there is nothing to exchange
   * \retval false otherwise.
   */
  virtual bool initializeCommunicationsMessages() = 0;

  /*! \brief Calculates communications.
   *
   * Assumes that the list of processors from which information is desired is in
   * \a recv_ranks.
   */
  virtual void initializeCommunicationsMessages(Int32ConstArrayView recv_ranks) = 0;

  //! Performs the exchange using the default options of ParallelExchangerOptions.
  virtual void processExchange() = 0;

  //! Performs the exchange using the options \a options
  virtual void processExchange(const ParallelExchangerOptions& options) = 0;

 public:

  virtual IParallelMng* parallelMng() const = 0;

  //! Number of processors to which we send
  virtual Integer nbSender() const = 0;
  //! List of ranks of processors to which we send
  virtual Int32ConstArrayView senderRanks() const = 0;
  //! Adds a processor to send to
  virtual void addSender(Int32 rank) = 0;
  //! Message intended for the \a i-th processor
  virtual ISerializeMessage* messageToSend(Integer i) = 0;

  //! Number of processors from which we will receive messages
  virtual Integer nbReceiver() const = 0;
  //! List of ranks of processors from which we will receive messages
  virtual Int32ConstArrayView receiverRanks() = 0;
  //! Message received from the \a i-th processor
  virtual ISerializeMessage* messageToReceive(Integer i) = 0;

  //! Sets the exchange mode.
  [[deprecated("Y2021: Use ParallelExchangerOptions::setExchangeMode()")]]
  virtual void setExchangeMode(eExchangeMode mode) = 0;
  //! Specified exchange mode
  [[deprecated("Y2021: Use ParallelExchangerOptions::exchangeMode()")]]
  virtual eExchangeMode exchangeMode() const = 0;

  //! Sets the verbosity level
  virtual void setVerbosityLevel(Int32 v) = 0;
  //! Verbosity level
  virtual Int32 verbosityLevel() const = 0;

  //! Sets the instance name. This name is used during prints
  virtual void setName(const String& name) = 0;
  //! Instance name
  virtual String name() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
