// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiControlDispatcher.h                                      (C) 2000-2025 */
/*                                                                           */
/* Manage Control/Utility parallel messages for MPI.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPICONTROLDISPATCHER_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPICONTROLDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing/IControlDispatcher.h"
#include "arccore/base/NotImplementedException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiControlDispatcher
: public IControlDispatcher
{
 public:

  MpiControlDispatcher(MpiAdapter* adapter);

 public:

  void waitAllRequests(ArrayView<Request> requests) override;
  void waitSomeRequests(ArrayView<Request> requests, ArrayView<bool> indexes,
                        bool is_non_blocking) override;
  IMessagePassingMng* commSplit(bool keep) override;
  void barrier() override;
  Request nonBlockingBarrier() override;
  MessageId probe(const PointToPointMessageInfo& message) override;
  MessageSourceInfo legacyProbe(const PointToPointMessageInfo& message) override;
  Ref<IRequestList> createRequestListRef() override
  {
    ARCCORE_THROW(NotImplementedException,"");
  }
  IProfiler* profiler() const override;
  void setProfiler(IProfiler* p) override;

 public:

  MpiAdapter* adapter() const { return m_adapter; }

 private:

  MpiAdapter* m_adapter;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
