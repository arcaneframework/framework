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
/* MpiRequest.h                                                (C) 2000-2020 */
/*                                                                           */
/* Spécialisation de 'Request' pour MPI.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPIREQUEST_H
#define ARCCORE_MESSAGEPASSINGMPI_MPIREQUEST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

#include "arccore/message_passing/Request.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{
class MpiAdapter;
/*!
 * \brief Spécialisation MPI d'une 'Request'.
 *
 * Cette classe permet de garantir qu'une requête MPI est bien construite
 * à partir d'une MPI_Request.
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiRequest
: public Request
{
 public:

  MpiRequest() = default;
  MpiRequest(int ret_value,MpiAdapter* creator,MPI_Request mpi_request)
  : Request(ret_value,creator,mpi_request){}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

