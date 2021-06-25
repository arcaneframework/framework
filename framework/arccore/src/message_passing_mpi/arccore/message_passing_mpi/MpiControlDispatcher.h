// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiControlDispatcher.h                                      (C) 2000-2020 */
/*                                                                           */
/* Manage Control/Utility parallel messages for MPI.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPICONTROLDISPATCHER_H
#define ARCCORE_MESSAGEPASSINGMPI_MPICONTROLDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing/IControlDispatcher.h"
#include "arccore/base/NotImplementedException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{
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
  MessageId probe(const PointToPointMessageInfo& message) override;
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
