// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PrivateVariableArrayTpl.cc                                  (C) 2000-2016 */
/*                                                                           */
/* Instanciation des classes templates communes des variables du maillage.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/PrivateVariableArrayT.H"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class PrivateVariableArrayT<Byte>; 
template class PrivateVariableArrayT<Int16>;
template class PrivateVariableArrayT<Int32>;  
template class PrivateVariableArrayT<Int64>; 
template class PrivateVariableArrayT<Real>; 
template class PrivateVariableArrayT<Real2>; 
template class PrivateVariableArrayT<Real2x2>; 
template class PrivateVariableArrayT<Real3>; 
template class PrivateVariableArrayT<Real3x3>; 

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
