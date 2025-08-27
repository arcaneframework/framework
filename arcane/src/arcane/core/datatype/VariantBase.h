// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariantBase.h                                               (C) 2000-2006 */
/*                                                                           */
/* Classe de base pour les types polymorphes.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPE_VARIANTBASE_H
#define ARCANE_DATATYPE_VARIANTBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/datatype/DataTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base pour les types polymorphes.
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
  : m_dim(dim), m_type(atype) {}
  virtual ~VariantBase() {}
 public:
  /*!
    \brief Dimension du variant.
    
    Les valeurs possibles sont les suivantes:
    - 0 pour un scalaire,.
    - 1 pour un tableau mono-dim ou variable scalaire du maillage.
  */
  Integer dimension() const { return m_dim; }
  eType type() const { return m_type; }
  const char* typeName() const { return typeName(m_type); }
  static const char* typeName(eType type);
  static eType fromDataType(eDataType type);

 protected:
  Integer m_dim; //!< dimension du variant.
  eType m_type; //!< Type garanti valide de la valeur.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

