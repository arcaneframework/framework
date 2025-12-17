// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingMng.h                                         (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire des échanges de messages.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGEPASSINGMNG_H
#define ARCCORE_MESSAGEPASSING_MESSAGEPASSINGMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arccore/message_passing/IMessagePassingMng.h"
#include "arccore/message_passing/Communicator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire des échanges de messages.
 *
 * Les instances de ces classes doivent être détruites via la méthode
 * mpDelete().
 */
class ARCCORE_MESSAGEPASSING_EXPORT MessagePassingMng
: public ReferenceCounterImpl
, public IMessagePassingMng
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  MessagePassingMng(Int32 comm_rank,Int32 comm_size,IDispatchers* d);
  // TODO: Rendre obsolète fin 2022: [[deprecated("Use mpDelete() instead")]]
  ~MessagePassingMng() override;

 public:

  Int32 commRank() const override { return m_comm_rank; }
  Int32 commSize() const override { return m_comm_size; }
  IDispatchers* dispatchers() override;
  ITimeMetricCollector* timeMetricCollector() const override;
  Communicator communicator() const override;

 public:

  void setTimeMetricCollector(ITimeMetricCollector* c);
  void setCommunicator(Communicator c);

 private:

  Int32 m_comm_rank = A_NULL_RANK;
  Int32 m_comm_size = A_NULL_RANK;
  IDispatchers* m_dispatchers = nullptr;
  ITimeMetricCollector* m_time_metric_collector = nullptr;
  Communicator m_communicator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
