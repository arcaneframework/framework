// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialSynchronizeBuffer.h                            (C) 2000-2022 */
/*                                                                           */
/* Interface des buffers pour la synchronisation de variables matériaux.     */
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface des buffers pour la synchronisation de variables matériaux.
 */
class ARCANE_MATERIALS_EXPORT IMeshMaterialSynchronizeBuffer
{
 public:

  virtual ~IMeshMaterialSynchronizeBuffer(){}

 public:

  //! Nombre de rangs
  virtual Int32 nbRank() const =0;

  //! Positionne le nombre de rangs. Cela invalide les buffers d'envoi et de réception
  virtual void setNbRank(Int32 nb_rank) =0;

  //! Buffer d'envoi pour le rang \a rank
  virtual Span<Byte> sendBuffer(Int32 rank) =0;

  //! Positionne le nombre d'éléments pour le buffer d'envoi du rang \a rank
  virtual void resizeSendBuffer(Int32 rank,Int64 new_size) =0;

  //! Buffer d'envoi pour le rang \a rank
  virtual Span<Byte> receiveBuffer(Int32 rank) =0;

  //! Positionne le nombre d'éléments pour le buffer de réceptions du rang \a rank
  virtual void resizeReceiveBuffer(Int32 rank,Int64 new_size) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_MATERIALS_EXPORT Ref<IMeshMaterialSynchronizeBuffer>
makeMeshMaterialSynchronizeBufferRef();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

