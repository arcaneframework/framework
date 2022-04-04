// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializeBuffer.cc                                          (C) 2000-2020 */
/*                                                                           */
/* Tampon de serialisation.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/SerializeBuffer.h"
#include "arcane/IParallelMng.h"
#include "arcane/utils/ITraceMng.h"

// TEMPORAIRE
#include "arccore/serialize/BasicSerializerInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializeBuffer::
allGather(IParallelMng* pm,const SerializeBuffer& send_serializer)
{
  const SerializeBuffer* sbuf = &send_serializer;
  SerializeBuffer* recv_buf = this;
  SerializeBuffer::Impl2* sbuf_p2 = sbuf->m_p2;
  SerializeBuffer::Impl2* recv_p2 = recv_buf->m_p2;

  Span<const Real> send_real = sbuf_p2->realBytes();
  Span<const Int16> send_int16 = sbuf_p2->int16Bytes();
  Span<const Int32> send_int32 = sbuf_p2->int32Bytes();
  Span<const Int64> send_int64 = sbuf_p2->int64Bytes();
  Span<const Byte> send_byte = sbuf_p2->byteBytes();

  Int64 sizes[5];
  sizes[0] = send_real.size();
  sizes[1] = send_int16.size();
  sizes[2] = send_int32.size();
  sizes[3] = send_int64.size();
  sizes[4] = send_byte.size();
  
  ITraceMng* msg = pm->traceMng();
  msg->info(4) << "SBUF_GATHER SIZE real=" << sizes[0] << " int16=" << sizes[1]
               << " int32=" << sizes[2] << " int64=" << sizes[3]
               << " bytes=" << sizes[4];

 pm->reduce(Parallel::ReduceSum,Int64ArrayView(5,sizes));

  Int64 recv_nb_real = sizes[0];
  Int64 recv_nb_int16 = sizes[1];
  Int64 recv_nb_int32 = sizes[2];
  Int64 recv_nb_int64 = sizes[3];
  Int64 recv_nb_byte = sizes[4];

  recv_p2->allocateBuffer(recv_nb_real,recv_nb_int16,recv_nb_int32,recv_nb_int64,recv_nb_byte);
  auto recv_p = recv_buf->_p();
  {
    RealUniqueArray real_buf;
    pm->allGatherVariable(send_real.smallView(),real_buf);
    recv_p->getRealBuffer().copy(real_buf);
  }

  {
    Int32UniqueArray int32_buf;
    pm->allGatherVariable(send_int32.smallView(),int32_buf);
    recv_p->getInt32Buffer().copy(int32_buf);
  }

  {
    Int16UniqueArray int16_buf;
    pm->allGatherVariable(send_int16.smallView(),int16_buf);
    recv_p->getInt16Buffer().copy(int16_buf);
  }

  {
    Int64UniqueArray int64_buf;
    pm->allGatherVariable(send_int64.smallView(),int64_buf);
    recv_p->getInt64Buffer().copy(int64_buf);
  }

  {
    ByteUniqueArray byte_buf;
    pm->allGatherVariable(send_byte.smallView(),byte_buf);
    recv_p->getByteBuffer().copy(byte_buf);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
