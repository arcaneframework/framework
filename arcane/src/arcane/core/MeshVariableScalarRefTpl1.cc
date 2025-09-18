// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableScalarRefTpl1.cc                                (C) 2000-2025 */
/*                                                                           */
/* Instanciation des classes templates des variables du maillage.            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshVariableScalarRef.inst.h"
#include "arcane/core/MeshPartialVariableScalarRef.inst.h"
#include "arcane/core/VariableFactoryRegisterer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemNumericOperation<Real>::
add(VarType& out,const VarType& v,const ItemGroup& group)
{
  ENUMERATE_ITEM(iitem,group){
    out[iitem] += v[iitem];
  }
}

void ItemNumericOperation<Real>::
sub(VarType& out,const VarType& v,const ItemGroup& group)
{
  ENUMERATE_ITEM(iitem,group){
    out[iitem] -= v[iitem];
  }
}

void ItemNumericOperation<Real>::
mult(VarType& out,const VarType& v,const ItemGroup& group)
{
  ENUMERATE_ITEM(iitem,group){
    out[iitem] *= v[iitem];
  }
}

void ItemNumericOperation<Real>::
mult(VarType& out,Real v,const ItemGroup& group)
{
  ENUMERATE_ITEM(iitem,group){
    out[iitem] *= v;
  }
}

void ItemNumericOperation<Real>::
power(VarType& out,Real v,const ItemGroup& group)
{
  ENUMERATE_ITEM(iitem,group){
    out[iitem] = math::pow(out[iitem],v);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real2);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real2x2);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real3);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real3x3);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
