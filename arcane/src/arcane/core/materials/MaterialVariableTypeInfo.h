﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialVariableTypeInfo.h                                  (C) 2000-2022 */
/*                                                                           */
/* Informations caractérisants le type d'une variable matériaux.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_MATERIALVARIABLETYPEINFO_H
#define ARCANE_CORE_MATERIALS_MATERIALVARIABLETYPEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/VariableTypeInfo.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations caractérisant le type d'une variable matériau.
 *
 * Les instances de cette classes peuvent être utilisées dans les
 * constructeurs statiques. Pour éviter tout problème cette classe ne doit
 * pas utiliser d'allocation dynamique.
 */
class ARCANE_CORE_EXPORT MaterialVariableTypeInfo
{
 public:

  constexpr MaterialVariableTypeInfo(eItemKind item_kind, eDataType data_type, Integer dimension)
  : m_variable_type_info(item_kind, data_type, dimension, 0, false)
  {}

 public:

  //! Type d'entité de maillage
  constexpr eItemKind itemKind() const { return m_variable_type_info.itemKind(); }
  //! Dimension
  constexpr Integer dimension() const { return m_variable_type_info.dimension(); }
  //! Type des données de la variable
  constexpr eDataType dataType() const { return m_variable_type_info.dataType(); }
  //! Nom complet du type de la variable
  String fullName() const;

 private:

  VariableTypeInfo m_variable_type_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
