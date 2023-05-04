// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariableInternal.h                             (C) 2000-2023 */
/*                                                                           */
/* API interne Arcane de 'IMeshMaterialVariable'.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_INTERNAL_IMESHMATERIALVARIABLEINTERNAL_H
#define ARCANE_CORE_MATERIALS_INTERNAL_IMESHMATERIALVARIABLEINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief API interne Arcane de 'IMeshMaterialVariable'.
 */
class ARCANE_CORE_EXPORT IMeshMaterialVariableInternal
{
 public:

  virtual ~IMeshMaterialVariableInternal() = default;

 public:

  /*!
   *\brief Taille en octet pour conserver une valeur de la variable.
   *
   * Pour une variable scalaire, il s'agit de la taille du type de donnée associé.
   * Pour une variable tableau, il s'agit de la taille du type de donnée 
   * multiplié pour le nombre d'éléments du tableau.
   */
  virtual Int32 dataTypeSize() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
