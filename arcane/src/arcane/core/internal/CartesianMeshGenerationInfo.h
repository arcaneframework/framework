// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshGenerationInfo.h                               (C) 2000-2022 */
/*                                                                           */
/* Informations sur la génération des maillages cartésiens.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_CARTESIANMESHGENERATIONINFO_H
#define ARCANE_CORE_INTERNAL_CARTESIANMESHGENERATIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Real3.h"

#include "arcane/core/ICartesianMeshGenerationInfo.h"

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

  Int64 globalNbCell() const override { return m_global_nb_cell; }
  Int64ConstArrayView globalNbCells() const override { return m_global_nb_cells; }
  Int32ConstArrayView subDomainOffsets() const override { return m_sub_domain_offsets; }
  Int32ConstArrayView nbSubDomains() const override { return m_nb_sub_domains; }
  Int32ConstArrayView ownNbCells() const override { return m_own_nb_cells; };
  Int64ConstArrayView ownCellOffsets() const override { return m_own_cell_offsets; };
  Int64 firstOwnCellUniqueId() const override { return m_first_own_cell_unique_id; }
  Real3 globalOrigin() const override { return m_global_origin; };
  Real3 globalLength() const override { return m_global_length; };

 public:

  void setOwnCellOffsets(Int64 x,Int64 y,Int64 z) override;
  void setGlobalNbCells(Int64 x,Int64 y,Int64 z) override;
  void setSubDomainOffsets(Int32 x,Int32 y,Int32 z) override;
  void setNbSubDomains(Int32 x,Int32 y,Int32 z) override;
  void setOwnNbCells(Int32 x,Int32 y,Int32 z) override;
  void setFirstOwnCellUniqueId(Int64 uid) override;
  void setGlobalOrigin(Real3 pos) override;
  void setGlobalLength(Real3 length) override;

 private:

  IMesh* m_mesh;
  Int32 m_mesh_dimension = -1;
  Int64 m_global_nb_cell = 0;

  Int64ArrayView m_global_nb_cells;
  Int32ArrayView m_sub_domain_offsets;
  Int32ArrayView m_nb_sub_domains;
  Int32ArrayView m_own_nb_cells;
  Int64ArrayView m_own_cell_offsets;

  Int64 m_global_nb_cell_ptr[NB_DIM];
  Int32 m_sub_domain_offset_ptr[NB_DIM];
  Int32 m_nb_sub_domain_ptr[NB_DIM];
  Int32 m_own_nb_cell_ptr[NB_DIM];
  Int64 m_own_cell_offset_ptr[NB_DIM];
  Real3 m_global_origin;
  Real3 m_global_length;

  Int64 m_first_own_cell_unique_id = -1;

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
