// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellToAllEnvCellConverter.h                                 (C) 2000-2024 */
/*                                                                           */
/* Conversion of 'Cell' to 'AllEnvCell'.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_CELLTOALLENVCELLCONVERTER_H
#define ARCANE_CORE_MATERIALS_CELLTOALLENVCELLCONVERTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IMesh.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"
#include "arcane/core/materials/MatItem.h"
#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/MatItemEnumerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Conversion of 'Cell' to 'AllEnvCell'.

 Instances of this class allow converting a \a Cell mesh
 into an \a AllEnvCell mesh to obtain material information.
 
 An instance is created via the material manager:
 \code
 * IMeshMaterialMng* mm = ...;
 * CellToAllEnvCellConverter all_env_cell_converter(mm);
 \endcode

 The creation cost is low, equivalent to a virtual function call. It is
 therefore not negligible, and it is preferable not to construct an instance
 inside loops over entities, for example, but outside them.

 Once the instance is created, it is then possible to use
 the [] operator (operator[]()) to perform the conversion:
 
 \code
 * CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);
 * Cell cell = ...;
 * ENUMERATE_FACE(iface,allFaces()){
 *   Face face = *iface;
 *   Cell back_cell = face.backCell()
 *   AllEnvCell back_all_env_cell = all_env_cell_converter[back_cell];
 *   info() << "nb env=" << back_all_env_cell.nbEnvironment();
 * }
 \endcode
 
 \warning Instances of this class are invalidated if the list of
 material or environment meshes changes. In this case, the object must be rebuilt:

 \code
 * all_env_cell_converter = CellToAllEnvCellConverter(m_material_mng);
 \endcode
 */
class CellToAllEnvCellConverter
{
  friend class MeshMaterialMng;

 public:

  explicit CellToAllEnvCellConverter(IMeshMaterialMng* mm)
  {
    *this = mm->cellToAllEnvCellConverter();
  }

 private:

  explicit CellToAllEnvCellConverter(ComponentItemSharedInfo* shared_info)
  : m_shared_info(shared_info)
  {
  }

 public:

  //! Converts a \a Cell mesh to an \a AllEnvCell mesh
  AllEnvCell operator[](Cell c)
  {
    return operator[](CellLocalId(c));
  }

  //! Converts a \a CellLocalId mesh to an \a AllEnvCell mesh
  ARCCORE_HOST_DEVICE AllEnvCell operator[](CellLocalId c) const
  {
    return AllEnvCell(m_shared_info->_item(ConstituentItemIndex(c.localId())));
  }

 private:

  ComponentItemSharedInfo* m_shared_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
