// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialSynchronizeBuffer.h                            (C) 2000-2023 */
/*                                                                           */
/* Interface for buffers for material variable synchronization.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_IMESHMATERIALSYNCHRONIZEBUFFER_H
#define ARCANE_MATERIALS_IMESHMATERIALSYNCHRONIZEBUFFER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Ref.h"

#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
/*
 * TODO: This interface could be used outside of materials.
 *       Look into how to make it generic.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for buffers for material variable synchronization.
 *
 * To use instances of this interface, proceed as follows:
 * 1. Set the number of ranks via setNbRank().
 * 2. For each buffer, call setSendBufferSize() and setReceiveBufferSize()
 *    to indicate the number of elements in each buffer.
 * 3. Call allocate() to allocate the buffers.
 * 4. Retrieve views on the buffers via sendBuffer() or receiveBuffer().
 */
class ARCANE_MATERIALS_EXPORT IMeshMaterialSynchronizeBuffer
{
 public:

  virtual ~IMeshMaterialSynchronizeBuffer() {}

 public:

  //! Number of ranks
  virtual Int32 nbRank() const = 0;

  //! Sets the number of ranks. This invalidates the send and receive buffers
  virtual void setNbRank(Int32 nb_rank) = 0;

  //! Send buffer for the i-th buffer
  virtual Span<Byte> sendBuffer(Int32 i) = 0;

  //! Sets the number of elements for the i-th send buffer
  virtual void setSendBufferSize(Int32 i, Int32 new_size) = 0;

  //! Send buffer for the i-th buffer
  virtual Span<Byte> receiveBuffer(Int32 i) = 0;

  //! Sets the number of elements for the i-th receive buffer
  virtual void setReceiveBufferSize(Int32 i, Int32 new_size) = 0;

  //! Allocates memory for the buffers
  virtual void allocate() = 0;

  //! Total size allocated for the buffers
  virtual Int64 totalSize() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{
  extern "C++" ARCANE_MATERIALS_EXPORT Ref<IMeshMaterialSynchronizeBuffer>
  makeMultiBufferMeshMaterialSynchronizeBufferRef();
  extern "C++" ARCANE_MATERIALS_EXPORT Ref<IMeshMaterialSynchronizeBuffer>
  makeMultiBufferMeshMaterialSynchronizeBufferRef(eMemoryRessource mem);
  extern "C++" ARCANE_MATERIALS_EXPORT Ref<IMeshMaterialSynchronizeBuffer>
  makeOneBufferMeshMaterialSynchronizeBufferRef(eMemoryRessource mem);
} // namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
