// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentListPrinter.cc                                   (C) 2000-2023 */
/*                                                                           */
/* Fonctions utilitaires d'affichage des constituants.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/ConstituentListPrinter.h"

#include "arcane/materials/internal/MeshMaterialMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentListPrinter::
ConstituentListPrinter(MeshMaterialMng* mm)
: TraceAccessor(mm->traceMng())
, m_material_mng(mm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentListPrinter::
print()
{
  IMesh* mesh = m_material_mng->mesh();
  _printConstituentsPerCell(mesh->allCells().view());
  _printConstituents();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentListPrinter::
_printConstituentsPerCell(ItemVectorView items)
{
  info() << "ConstituentsPerCell:";
  CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);
  ENUMERATE_ (Cell, icell, items) {
    AllEnvCell all_env_cell = all_env_cell_converter[icell];
    Cell global_cell = all_env_cell.globalCell();
    info() << "Cell=" << global_cell.uniqueId();
    ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
      EnvCell ec = *ienvcell;
      info() << " EnvCell mv=" << ec._varIndex()
             << " env=" << ec.component()->name();
      ENUMERATE_CELL_MATCELL (imatcell, (*ienvcell)) {
        MatCell mc = *imatcell;
        info() << "  MatCell mv=" << mc._varIndex()
               << " mat=" << mc.component()->name();
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentListPrinter::
_printConstituents()
{
  info() << "Constituents:";
  ENUMERATE_ENV (ienv, m_material_mng) {
    IMeshEnvironment* env = *ienv;
    info() << "ENV name=" << env->name();
    ENUMERATE_ENVCELL (icell, env) {
      EnvCell ev = *icell;
      info() << "EnvCell mv=" << ev._varIndex();
    }
    ENUMERATE_MAT (imat, env) {
      Materials::IMeshMaterial* mat = *imat;
      info() << "MAT name=" << mat->name();
      ENUMERATE_MATCELL (icell, mat) {
        MatCell mc = *icell;
        info() << "MatCell mv=" << mc._varIndex();
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
