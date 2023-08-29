// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialSynchronizeBuffer.cc                            (C) 2000-2023 */
/*                                                                           */
/* Gestion des buffers pour la synchronisation de variables matériaux.       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/IMeshMaterialSynchronizeBuffer.h"

#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IMemoryRessourceMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MultiBufferMeshMaterialSynchronizeBuffer
: public IMeshMaterialSynchronizeBuffer
{
 public:

  struct BufferInfo
  {
    BufferInfo(IMemoryAllocator* allocator)
    : m_send_buffer(allocator)
    , m_receive_buffer(allocator)
    {
    }

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

  MultiBufferMeshMaterialSynchronizeBuffer(IMemoryAllocator* allocator)
  : m_default_buffer_info(allocator)
  {
  }

  Int32 nbRank() const override { return m_nb_rank; }
  void setNbRank(Int32 nb_rank) override
  {
    m_nb_rank = nb_rank;
    m_buffer_infos.resize(nb_rank,m_default_buffer_info);
    for (auto& x : m_buffer_infos)
      x.reset();
  }
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
  }
  void setReceiveBufferSize(Int32 index, Int32 new_size) override
  {
    m_buffer_infos[index].m_receive_size = new_size;
  }
  void allocate() override
  {
    m_total_size = 0;
    for (auto& x : m_buffer_infos) {
      x.m_send_buffer.resize(x.m_send_size);
      x.m_receive_buffer.resize(x.m_receive_size);
      m_total_size += x.m_send_size + x.m_receive_size;
    }
  }
  Int64 totalSize() const override { return m_total_size; }

 public:

  Int32 m_nb_rank = 0;
  Int64 m_total_size = 0;
  BufferInfo m_default_buffer_info;
  UniqueArray<BufferInfo> m_buffer_infos;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OneBufferMeshMaterialSynchronizeBuffer
: public IMeshMaterialSynchronizeBuffer
{
 public:

  struct BufferInfo
  {
    void reset()
    {
      m_send_size = 0;
      m_receive_size = 0;
      m_send_index = 0;
      m_receive_index = 0;
    }

    Span<Byte> sendBuffer(Span<Byte> full_buffer) const
    {
      return full_buffer.subspan(m_send_index,m_send_size);
    }
    Span<Byte> receiveBuffer(Span<Byte> full_buffer) const
    {
      return full_buffer.subspan(m_receive_index,m_receive_size);
    }

    Int32 m_send_size = 0;
    Int32 m_receive_size = 0;
    Int64 m_send_index = 0;
    Int64 m_receive_index = 0;
  };

 public:

  explicit OneBufferMeshMaterialSynchronizeBuffer(IMemoryAllocator* allocator)
  : m_buffer(allocator){}

 public:
  Int32 nbRank() const override { return m_nb_rank; }
  void setNbRank(Int32 nb_rank) override
  {
    m_nb_rank = nb_rank;
    m_buffer_infos.resize(nb_rank);
    for (auto& x : m_buffer_infos)
      x.reset();
  }
  Span<Byte> sendBuffer(Int32 index) override
  {
    return m_buffer_infos[index].sendBuffer(m_buffer);
  }
  void setSendBufferSize(Int32 index, Int32 new_size) override
  {
    m_buffer_infos[index].m_send_size = new_size;
  }
  Span<Byte> receiveBuffer(Int32 index) override
  {
    return m_buffer_infos[index].receiveBuffer(m_buffer);
  }
  void setReceiveBufferSize(Int32 index, Int32 new_size) override
  {
    m_buffer_infos[index].m_receive_size = new_size;
  }
  void allocate() override
  {
    Int64 total_send_size = 0;
    Int64 total_receive_size = 0;
    for (auto& x : m_buffer_infos) {
      total_send_size += x.m_send_size;
      total_receive_size += x.m_receive_size;
    }
    m_buffer.resize(total_send_size+total_receive_size);
    Int64 send_index = 0;
    Int64 receive_index = total_send_size;
    for (auto& x : m_buffer_infos) {
      x.m_send_index = send_index;
      x.m_receive_index = receive_index;
      send_index += x.m_send_size;
      receive_index += x.m_receive_size;
    }
  }
  Int64 totalSize() const override { return m_buffer.largeSize(); }

 public:

  Int32 m_nb_rank = 0;
  UniqueArray<BufferInfo> m_buffer_infos;
  UniqueArray<Byte> m_buffer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{
extern "C++" ARCANE_MATERIALS_EXPORT Ref<IMeshMaterialSynchronizeBuffer>
makeMultiBufferMeshMaterialSynchronizeBufferRef(eMemoryRessource memory_ressource)
{
  auto* a = platform::getDataMemoryRessourceMng()->getAllocator(memory_ressource);
  auto* v = new MultiBufferMeshMaterialSynchronizeBuffer(a);
  return makeRef<IMeshMaterialSynchronizeBuffer>(v);
}

extern "C++" ARCANE_MATERIALS_EXPORT Ref<IMeshMaterialSynchronizeBuffer>
makeMultiBufferMeshMaterialSynchronizeBufferRef()
{
  return makeMultiBufferMeshMaterialSynchronizeBufferRef(eMemoryRessource::Host);
}

extern "C++" ARCANE_MATERIALS_EXPORT Ref<IMeshMaterialSynchronizeBuffer>
makeOneBufferMeshMaterialSynchronizeBufferRef(eMemoryRessource memory_ressource)
{
  auto* a = platform::getDataMemoryRessourceMng()->getAllocator(memory_ressource);
  auto* v = new OneBufferMeshMaterialSynchronizeBuffer(a);
  return makeRef<IMeshMaterialSynchronizeBuffer>(v);
}
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
