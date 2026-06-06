// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariantBase.h                                               (C) 2000-2025 */
/*                                                                           */
/* Base class for polymorphic types.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_DATATYPE_VARIANTBASE_H
#define ARCANE_CORE_DATATYPE_VARIANTBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/datatype/DataTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Base class for polymorphic types.
 */
class ARCANE_DATATYPE_EXPORT VariantBase
{
 public:

  enum eType
  {
    TReal = 0,
    TInt32 = 1,
    TInt64 = 2,
    TBool = 3,
    TString = 4,
    TReal2 = 5,
    TReal3 = 6,
    TReal2x2 = 7,
    TReal3x3 = 8,
    TUnknown = 9
  };

 public:

  VariantBase(Integer dim, eType atype)
  : m_dim(dim)
  , m_type(atype)
  {}
  virtual ~VariantBase() {}

 public:

  /*!
    \brief Variant dimension.
    
    The possible values are as follows:
    - 0 for a scalar.
    - 1 for a mono-dimensional array or scalar variable of the mesh.
  */
  Integer dimension() const { return m_dim; }
  eType type() const { return m_type; }
  const char* typeName() const { return typeName(m_type); }
  static const char* typeName(eType type);
  static eType fromDataType(eDataType type);

 protected:

  Integer m_dim; //!< variant dimension.
  eType m_type; //!< Guaranteed valid type of the value.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
