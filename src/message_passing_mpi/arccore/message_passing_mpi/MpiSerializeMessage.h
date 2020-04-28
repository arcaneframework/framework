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
/* MpiSerializeMessage.h                                       (C) 2000-2020 */
/*                                                                           */
/* Encapsulation de ISerializeMessage pour MPI.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPISERIALIZEMESSAGE_H
#define ARCCORE_MESSAGEPASSINGMPI_MPISERIALIZEMESSAGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/serialize/SerializeGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiSerializeMessage
{
 public:
  MpiSerializeMessage(ISerializeMessage* message,Integer index);
  ~MpiSerializeMessage();
 public:
  ISerializeMessage* message() const { return m_message; }
  BasicSerializer* serializeBuffer() const { return m_serialize_buffer; }
  Integer messageNumber() const { return m_message_number; }
  void incrementMessageNumber() { ++m_message_number; }
 private:
  ISerializeMessage* m_message;
  BasicSerializer* m_serialize_buffer;
  //! Index du message
  Integer m_message_index;
  //! Numéro du message recu (0 pour le premier)
  Integer m_message_number;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

