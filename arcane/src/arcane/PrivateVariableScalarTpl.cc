// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PrivateVariableScalarTpl.cc                                 (C) 2000-2011 */
/*                                                                           */
/* Instanciation des classes templates communes des variables du maillage.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/PrivateVariableScalarT.H"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class PrivateVariableScalarT<Byte>; 
template class PrivateVariableScalarT<Int16>;
template class PrivateVariableScalarT<Int32>;
template class PrivateVariableScalarT<Int64>;
template class PrivateVariableScalarT<Real>; 
template class PrivateVariableScalarT<Real2>; 
template class PrivateVariableScalarT<Real2x2>; 
template class PrivateVariableScalarT<Real3>; 
template class PrivateVariableScalarT<Real3x3>; 

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
