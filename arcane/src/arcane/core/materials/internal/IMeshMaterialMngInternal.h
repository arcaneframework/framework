// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialMngInternal.h                                  (C) 2000-2023 */
/*                                                                           */
/* API interne Arcane de 'IMeshMaterialMng'.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_INTERNAL_IMESHMATERIALMNGINTERNAL_H
#define ARCANE_CORE_MATERIALS_INTERNAL_IMESHMATERIALMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief API interne Arcane de 'IMeshMaterialMng'.
 */
class ARCANE_CORE_EXPORT IMeshMaterialMngInternal
{
 public:

  virtual ~IMeshMaterialMngInternal() = default;

 public:

  /*!
   * \internal
   * \brief Renvoie la table de "connectivité" CellLocalId -> AllEnvCell
   * destinée à être utilisée dans un RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL
   * en conjonction de la macro ENUMERATE_CELL_ALLENVCELL
   */
  virtual AllCellToAllEnvCell* getAllCellToAllEnvCell() const =0;

  /*!
   * \internal
   * \brief Construit la table de "connectivité" CellLocalId -> AllEnvCell
   * destinée à être utilisée dans un RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL
   * en conjonction de la macro ENUMERATE_CELL_ALLENVCELL
   *
   * Si aucun allocateur n'est spécifié alors la méthode
   * platform::getDefaultDataAllocator() est utilisée
   */
  virtual void createAllCellToAllEnvCell(IMemoryAllocator* alloc=platform::getDefaultDataAllocator()) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
