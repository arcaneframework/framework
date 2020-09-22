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
/* StandaloneMpiMessagePassingMng.h                            (C) 2000-2020 */
/*                                                                           */
/* Version autonome de MpiMessagePassingMng.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_STANDALONEMPIMESSAGEPASSINGMNG_H
#define ARCCORE_MESSAGEPASSINGMPI_STANDALONEMPIMESSAGEPASSINGMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiMessagePassingMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Version autonome de MpiMessagePassingMng.
 *
 * La création se fait via la méthode statique create().
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT StandaloneMpiMessagePassingMng
: public MpiMessagePassingMng
{
  class Impl;

 private:

  StandaloneMpiMessagePassingMng(Impl* p);

 public:

  ~StandaloneMpiMessagePassingMng() override;

 public:

  //! Créé un gestionnaire associé au communicateur \a comm.
  static MpiMessagePassingMng* create(MPI_Comm comm, bool clean_comm=false);

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
