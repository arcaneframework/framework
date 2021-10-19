// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICartesianMeshGenerationInfo.h                              (C) 2000-2021 */
/*                                                                           */
/* Informations sur la génération des maillages cartésiens.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_ICARTESIANMESHGENERATIONINFO_H
#define ARCANE_CORE_INTERNAL_ICARTESIANMESHGENERATIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations sur la génération des maillages cartésiens.
 */
class ARCANE_CORE_EXPORT ICartesianMeshGenerationInfo
{
 public:

  static ICartesianMeshGenerationInfo* getReference(IMesh* mesh,bool create);

 public:

  virtual ~ICartesianMeshGenerationInfo() = default;

 public:

  virtual Int64ConstArrayView globalNbCell() const =0;
  virtual Int32ConstArrayView subDomainOffset() const =0;
  virtual Int32ConstArrayView ownNbCell() const =0;
  virtual Int64ConstArrayView ownCellOffset() const =0;

  virtual void setOwnCellOffset(Int64 x,Int64 y,Int64 z) =0;
  virtual void setGlobalNbCell(Int64 x,Int64 y,Int64 z) =0;
  virtual void setSubDomainOffset(Int32 x,Int32 y,Int32 z) =0;
  virtual void setOwnNbCell(Int32 x,Int32 y,Int32 z) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
