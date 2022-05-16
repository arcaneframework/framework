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

  Int32 nbRank() const override { return m_nb_rank; }
  void setNbRank(Int32 nb_rank) override;
  Span<Byte> sendBuffer(Int32 index) override
  {
    return m_send_buffers[index];
  }
  void resizeSendBuffer(Int32 index,Int64 new_size) override
  {
    m_send_buffers[index].resize(new_size);
  }
  Span<Byte> receiveBuffer(Int32 index) override
  {
    return m_receive_buffers[index];
  }
  void resizeReceiveBuffer(Int32 index,Int64 new_size) override
  {
    m_receive_buffers[index].resize(new_size);
  }

 public:

  Int32 m_nb_rank = 0;
  UniqueArray< UniqueArray<Byte> > m_send_buffers;
  UniqueArray< UniqueArray<Byte> > m_receive_buffers;
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
  m_send_buffers.resize(nb_rank);
  m_receive_buffers.resize(nb_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
