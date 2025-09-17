// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PrivateVariableScalarTpl.cc                                 (C) 2000-2025 */
/*                                                                           */
/* Instanciation des classes templates communes des variables du maillage.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumericTypes.h"
#include "arcane/core/PrivateVariableScalar.inst.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class PrivateVariableScalarT<Byte>; 
template class PrivateVariableScalarT<Int8>;
template class PrivateVariableScalarT<Int16>;
template class PrivateVariableScalarT<Int32>;
template class PrivateVariableScalarT<Int64>;
template class PrivateVariableScalarT<BFloat16>; 
template class PrivateVariableScalarT<Float16>; 
template class PrivateVariableScalarT<Float32>; 
template class PrivateVariableScalarT<Real>; 
template class PrivateVariableScalarT<Real2>; 
template class PrivateVariableScalarT<Real2x2>; 
template class PrivateVariableScalarT<Real3>; 
template class PrivateVariableScalarT<Real3x3>; 

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
