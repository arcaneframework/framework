// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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

#include "arcane/core/ArcaneTypes.h"

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

  virtual Int64 globalNbCell() const =0;
  virtual Int64ConstArrayView globalNbCells() const =0;
  virtual Int32ConstArrayView subDomainOffsets() const =0;
  virtual Int32ConstArrayView nbSubDomains() const =0;
  virtual Int32ConstArrayView ownNbCells() const =0;
  virtual Int64ConstArrayView ownCellOffsets() const =0;
  virtual Int64 firstOwnCellUniqueId() const =0;
  virtual Real3 globalOrigin() const =0;
  virtual Real3 globalLength() const =0;

  virtual void setOwnCellOffsets(Int64 x,Int64 y,Int64 z) =0;
  virtual void setGlobalNbCells(Int64 x,Int64 y,Int64 z) =0;
  virtual void setSubDomainOffsets(Int32 x,Int32 y,Int32 z) =0;
  virtual void setNbSubDomains(Int32 x,Int32 y,Int32 z) =0;
  virtual void setOwnNbCells(Int32 x,Int32 y,Int32 z) =0;
  virtual void setFirstOwnCellUniqueId(Int64 uid) =0;
  virtual void setGlobalOrigin(Real3 pos) =0;
  virtual void setGlobalLength(Real3 length) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
