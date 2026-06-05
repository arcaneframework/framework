// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AllCellToAllEnvCellConverter.h                              (C) 2000-2024 */
/*                                                                           */
/* Conversion of 'Cell' to 'AllEnvCell'.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_ALLCELLTOALLENVCELLCONVERTER_H
#define ARCANE_MATERIALS_ALLCELLTOALLENVCELLCONVERTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MaterialsGlobal.h"

#include "arcane/core/materials/MatItem.h"
#include "arcane/core/materials/MatItemEnumerator.h"
#include "arcane/core/materials/CellToAllEnvCellConverter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Connectivity table from 'Cell' to its 'AllEnvCell' intended
 *        for use on accelerator.
 *
 * Class that maintains the connectivity of all meshes
 * \a Cell to all their meshes \a AllEnvCell.
 *
 * An instance is created via the create() method.
 *
 * The initialization cost is high; memory must be allocated and structures filled. We iterate through all meshes and for each mesh, we call
 * the CellToAllEnvCellConverter.
 *
 * Once the instance is created, it must be updated every time that
 * the material/environment topology changes (which is also costly).
 *
 * This class is an internal class and should not be manipulated directly.
 * You must use the associated helpers in IMeshMaterialMng and
 * the CellToAllEnvCellAccessor class.
 */
class ARCANE_MATERIALS_EXPORT AllCellToAllEnvCell
{
  friend class CellToAllEnvCellAccessor;
  friend class CellToAllComponentCellEnumerator;
  friend AllCellToAllEnvCellContainer;

 private:

  //! Access method for the "connectivity" table cell -> all env cells
  ARCCORE_HOST_DEVICE Span<ComponentItemLocalId> operator[](Int32 cell_id) const
  {
    return m_allcell_allenvcell_ptr[cell_id];
  }

 private:

  Span<Span<ComponentItemLocalId>> m_allcell_allenvcell_ptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Encapsulation class to access the equivalent connectivity
 *        cell -> allenvcell. Intended to be used with the accelerator API
 *        via RUNCOMMAND_...
 * \note It has no inherent interest, other than forcing the user to create
 * this object in addition to calling a RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL and thus
 * to guarantee the copy of the AllCellToAllEnvCell pointer for the lambda to execute on
 * the accelerator
 */
class ARCANE_MATERIALS_EXPORT CellToAllEnvCellAccessor
{
  friend class CellToAllComponentCellEnumerator;

 public:

  using size_type = Span<ComponentItemLocalId>::size_type;

 public:

  CellToAllEnvCellAccessor() = default;
  explicit CellToAllEnvCellAccessor(const IMeshMaterialMng* mm);

  ARCCORE_HOST_DEVICE size_type nbEnvironment(Int32 cid) const
  {
    return m_cell_allenvcell[cid].size();
  }

 private:

  ARCCORE_HOST_DEVICE AllCellToAllEnvCell _getAllCellToAllEnvCell() const
  {
    return m_cell_allenvcell;
  }

 private:

  AllCellToAllEnvCell m_cell_allenvcell;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MATERIALS_EXPORT CellToAllComponentCellEnumerator
{
  friend class EnumeratorTracer;

 public:

  using index_type = Span<ComponentItemLocalId>::index_type;
  using size_type = Span<ComponentItemLocalId>::size_type;

 public:

  // The CPU version allows verification that initialization was done before ENUMERATE
  ARCCORE_HOST_DEVICE explicit CellToAllComponentCellEnumerator(Int32 cell_id, const CellToAllEnvCellAccessor& acc)
  {
    AllCellToAllEnvCell all_env_view = acc._getAllCellToAllEnvCell();
    m_ptr = all_env_view[cell_id];
  }
  ARCCORE_HOST_DEVICE void operator++()
  {
    ++m_index;
  }

  ARCCORE_HOST_DEVICE bool hasNext() const
  {
    return m_index < m_ptr.size();
  }

  ARCCORE_HOST_DEVICE ComponentItemLocalId operator*() const
  {
    return m_ptr[m_index];
  }

 private:

  index_type m_index = 0;
  Span<ComponentItemLocalId> m_ptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro to iterate over a group of meshes in order to iterate
 * over the allenvcells of each mesh.
 *
 * \note By forcing the use of CellToAllEnvCellAccessor in the macro,
 * we ensure the capture by copy of the AllCellToAllEnvCell pointer, allowing
 * the use of ENUMERATE_CELL_ALLENVCELL.
 *
 * TODO Very likely to be moved elsewhere if this prototype is kept
 */
#define RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL(cell_to_allenvcellaccessor, iter_name, cell_group) \
  A_FUNCINFO << cell_group << [=] ARCCORE_HOST_DEVICE(CellLocalId iter_name)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO Very likely to be moved elsewhere if this prototype is kept
#define A_ENUMERATE_CELL_ALLCOMPONENTCELL(_EnumeratorClassName, iname, cid, cell_to_allenvcellaccessor) \
  for (A_TRACE_COMPONENT(_EnumeratorClassName) iname(::Arcane::Materials::_EnumeratorClassName(cid, cell_to_allenvcellaccessor) A_TRACE_ENUMERATOR_WHERE); iname.hasNext(); ++iname)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro to iterate over all environments within a mesh.
 *        "Raw and lightweight" ENUMERATE_CELL_ENVCELL version, intended for
 *        use on accelerator, i.e. within a RUN_COMMAND...
 *
 * \param iname variable name (type MatVarIndex) allowing access to
 *              data.
 * \param cid mesh identifier (type Integer).
 * \param cell_to_allenvcellaccessor cell->allenvcell connectivity (type CellToAllEnvCellAccessor)
 */
// TODO Very likely to be moved elsewhere if this prototype is kept
#define ENUMERATE_CELL_ALLENVCELL(iname, cid, cell_to_allenvcellaccessor) \
  A_ENUMERATE_CELL_ALLCOMPONENTCELL(CellToAllComponentCellEnumerator, iname, cid, cell_to_allenvcellaccessor)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
