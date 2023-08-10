// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentListPrinter.h                                    (C) 2000-2023 */
/*                                                                           */
/* Fonctions utilitaires d'affichage des constituants.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_COMPONENTLISTPRINTER_H
#define ARCANE_MATERIALS_INTERNAL_COMPONENTLISTPRINTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/core/VariableTypes.h"

#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctions utilitaires d'affichage des constituants.
 */
class ARCANE_MATERIALS_EXPORT ConstituentListPrinter
: public TraceAccessor
{
 public:

  explicit ConstituentListPrinter(MeshMaterialMng* mm);

 public:

  ConstituentListPrinter(ConstituentListPrinter&&) = delete;
  ConstituentListPrinter(const ConstituentListPrinter&) = delete;
  ConstituentListPrinter& operator=(ConstituentListPrinter&&) = delete;
  ConstituentListPrinter& operator=(const ConstituentListPrinter&) = delete;

 public:

  void print();

 private:

  MeshMaterialMng* m_material_mng = nullptr;

 private:

  void _printConstituentsPerCell(ItemVectorView items);
  void _printConstituents();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
