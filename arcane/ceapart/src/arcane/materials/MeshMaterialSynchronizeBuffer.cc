// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialSynchronizeBuffer.cc                            (C) 2000-2022 */
/*                                                                           */
/* Gestion des buffers pour la synchronisation de variables matériaux.       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/IMeshMaterialSynchronizeBuffer.h"

#include "arcane/utils/UniqueArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialSynchronizeBuffer
: public IMeshMaterialSynchronizeBuffer
{
 public:

  struct BufferInfo
  {
    void reset()
    {
      m_send_size = 0;
      m_receive_size = 0;
      m_send_buffer.clear();
      m_receive_buffer.clear();
    }
    Int32 m_send_size = 0;
    Int32 m_receive_size = 0;
    UniqueArray<Byte> m_send_buffer;
    UniqueArray<Byte> m_receive_buffer;
  };

 public:

  Int32 nbRank() const override { return m_nb_rank; }
  void setNbRank(Int32 nb_rank) override;
  Span<Byte> sendBuffer(Int32 index) override
  {
    return m_buffer_infos[index].m_send_buffer;
  }
  void setSendBufferSize(Int32 index, Int32 new_size) override
  {
    m_buffer_infos[index].m_send_size = new_size;
  }
  Span<Byte> receiveBuffer(Int32 index) override
  {
    return m_buffer_infos[index].m_receive_buffer;
    ;
  }
  void setReceiveBufferSize(Int32 index, Int32 new_size) override
  {
    m_buffer_infos[index].m_receive_size = new_size;
  }
  void allocate() override;

 public:

  Int32 m_nb_rank = 0;
  UniqueArray<BufferInfo> m_buffer_infos;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_MATERIALS_EXPORT Ref<IMeshMaterialSynchronizeBuffer>
makeMeshMaterialSynchronizeBufferRef()
{
  IMeshMaterialSynchronizeBuffer* v = new MeshMaterialSynchronizeBuffer();
  return makeRef(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSynchronizeBuffer::
setNbRank(Int32 nb_rank)
{
  m_nb_rank = nb_rank;
  m_buffer_infos.resize(nb_rank);
  for (auto& x : m_buffer_infos)
    x.reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSynchronizeBuffer::
allocate()
{
  for (Integer i = 0; i < m_nb_rank; ++i) {
    m_buffer_infos[i].m_send_buffer.resize(m_buffer_infos[i].m_send_size);
    m_buffer_infos[i].m_receive_buffer.resize(m_buffer_infos[i].m_receive_size);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
