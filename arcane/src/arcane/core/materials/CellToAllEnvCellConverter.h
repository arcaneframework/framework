// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellToAllEnvCellConverter.h                                 (C) 2000-2024 */
/*                                                                           */
/* Conversion de 'Cell' en 'AllEnvCell'.                                     */
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
 * \brief Conversion de 'Cell' en 'AllEnvCell'.

 Les instances de cette classe permettent de convertir une maille \a Cell
 en une maille \a AllEnvCell afin d'avoir les infos sur les matériaux.
 
 La création d'une instance se fait via le gestionnaire de matériaux:
 \code
 * IMeshMaterialMng* mm = ...;
 * CellToAllEnvCellConverter all_env_cell_converter(mm);
 \endcode

 Le coût de la création est faible, équivalent à un appel de fonction
 virtuelle. Il n'est donc pas nul et il est préférable de ne pas construire
 d'instance dans les boucles sur les entités par exemple, mais au dehors.

 Une fois l'instance créée, il est ensuite possible d'utiliser
 l'opérateur [] (operator[]()) pour faire la conversion:
 
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
 
 \warning Les instances de cette classe sont invalidées si la liste des
 mailles matériaux ou milieu change. Dans ce cas, il faut
 refabriquer l'objet:

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

  //! Converti une maille \a Cell en maille \a AllEnvCell
  AllEnvCell operator[](Cell c)
  {
    return operator[](CellLocalId(c));
  }
  //! Converti une maille \a CellLocalId en maille \a AllEnvCell
  ARCCORE_HOST_DEVICE AllEnvCell operator[](CellLocalId c) const
  {
    //const ComponentItemInternal* p = &m_all_env_items_internal[c.localId()];
    // Pour les AllEnvCell, l'index dans la liste est le localid() de la maille.
    return AllEnvCell(m_shared_info->_item(ConstituentItemIndex(c.localId())));
  }

 private:

  ConstArrayView<ComponentItemInternal> m_all_env_items_internal;
  ComponentItemSharedInfo* m_shared_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

