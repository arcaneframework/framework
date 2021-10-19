// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshGenerationInfo.h                               (C) 2000-2021 */
/*                                                                           */
/* Informations sur la génération des maillages cartésiens.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_CARTESIANMESHGENERATIONINFO_H
#define ARCANE_CORE_INTERNAL_CARTESIANMESHGENERATIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/internal/ICartesianMeshGenerationInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations sur la génération des maillages cartésiens.
 */
class ARCANE_CORE_EXPORT CartesianMeshGenerationInfo
: public ICartesianMeshGenerationInfo
{
  static constexpr Int32 NB_DIM = 3;

 public:

  CartesianMeshGenerationInfo(IMesh* mesh);

 public:

  Int64ConstArrayView globalNbCell() const override { return m_global_nb_cell; }
  Int32ConstArrayView subDomainOffset() const override { return m_sub_domain_offset; }
  Int32ConstArrayView ownNbCell() const override { return m_own_nb_cell; };
  Int64ConstArrayView ownCellOffset() const override { return m_own_cell_offset; };

 public:

  void setOwnCellOffset(Int64 x,Int64 y,Int64 z) override;
  void setGlobalNbCell(Int64 x,Int64 y,Int64 z) override;
  void setSubDomainOffset(Int32 x,Int32 y,Int32 z) override;
  void setOwnNbCell(Int32 x,Int32 y,Int32 z) override;

 private:

  IMesh* m_mesh;

  Int64ArrayView m_global_nb_cell;
  Int32ArrayView m_sub_domain_offset;
  Int32ArrayView m_own_nb_cell;
  Int64ArrayView m_own_cell_offset;

  Int64 m_global_nb_cell_ptr[NB_DIM];
  Int32 m_sub_domain_offset_ptr[NB_DIM];
  Int32 m_own_nb_cell_ptr[NB_DIM];
  Int64 m_own_cell_offset_ptr[NB_DIM];

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
