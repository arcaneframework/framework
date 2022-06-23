// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingMng.h                                         (C) 2000-2022 */
/*                                                                           */
/* Gestionnaire des échanges de messages.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGEPASSINGMNG_H
#define ARCCORE_MESSAGEPASSING_MESSAGEPASSINGMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/IMessagePassingMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
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
: public IMessagePassingMng
{
 public:

  MessagePassingMng(Int32 comm_rank,Int32 comm_size,IDispatchers* d);
  // TODO: Rendre obsolète fin 2022: [[deprecated("Use mpDelete() instead")]]
  ~MessagePassingMng() override;

 public:

  Int32 commRank() const override { return m_comm_rank; }
  Int32 commSize() const override { return m_comm_size; }
  IDispatchers* dispatchers() override;
  ITimeMetricCollector* timeMetricCollector() const override;
  void setTimeMetricCollector(ITimeMetricCollector* c);

 private:

  Int32 m_comm_rank;
  Int32 m_comm_size;
  IDispatchers* m_dispatchers = nullptr;
  ITimeMetricCollector* m_time_metric_collector = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
