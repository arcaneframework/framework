// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialVariableTypeInfo.h                                  (C) 2000-2025 */
/*                                                                           */
/* Information characterizing the type of a material variable.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_MATERIALVARIABLETYPEINFO_H
#define ARCANE_CORE_MATERIALS_MATERIALVARIABLETYPEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableTypeInfo.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Information characterizing the type of a material variable.
 *
 * Instances of this class can be used in
 * static constructors. To avoid any issues, this class must not
 * use dynamic allocation.
 */
class ARCANE_CORE_EXPORT MaterialVariableTypeInfo
{
 public:

  constexpr MaterialVariableTypeInfo(eItemKind item_kind, eDataType data_type,
                                     Integer dimension, MatVarSpace space)
  : m_variable_type_info(item_kind, data_type, dimension, 0, false)
  , m_mat_var_space(space)
  {}

 public:

  //! Mesh entity type
  constexpr eItemKind itemKind() const { return m_variable_type_info.itemKind(); }
  //! Dimension
  constexpr Integer dimension() const { return m_variable_type_info.dimension(); }
  //! Data type of the variable
  constexpr eDataType dataType() const { return m_variable_type_info.dataType(); }
  //! Full name of the variable type
  String fullName() const;

 private:

  VariableTypeInfo m_variable_type_info;
  MatVarSpace m_mat_var_space;

 private:

  String _buildFullTypeName() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
