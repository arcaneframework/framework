// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariantBase.cc                                              (C) 2000-2004 */
/*                                                                           */
/* Type de base polymorphe pour les tableaux mono-dim (dimension 1).         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/datatype/VariantBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DATATYPE_EXPORT const char* VariantBase::
typeName(eType type)
{
  switch(type){
  case TReal: return "Real";
  case TInt32: return "Int32";
  case TInt64: return "Int64";
  case TBool: return "Bool";
  case TString: return "String";
  case TReal2: return "Real2";
  case TReal3: return "Real3";
  case TReal2x2: return "Real2x2";
  case TReal3x3: return "Real3x3";
  default:
    break;
  }
  return "Unknown";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariantBase::eType VariantBase::
fromDataType(eDataType type)
{
  switch(type){
  case DT_Real: return VariantBase::TReal;
  case DT_Int32: return VariantBase::TInt32;
  case DT_Int64: return VariantBase::TInt64;
  case DT_String: return VariantBase::TString;
  case DT_Real2: return VariantBase::TReal2;
  case DT_Real3: return VariantBase::TReal3;
  case DT_Real2x2: return VariantBase::TReal2x2;
  case DT_Real3x3: return VariantBase::TReal3x3;
  default:
    break;
  }
  return VariantBase::TUnknown;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
