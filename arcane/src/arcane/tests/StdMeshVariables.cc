// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdMeshVariables.cc                                         (C) 2000-2024 */
/*                                                                           */
/* Définition de variables du maillage pour des tests.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/Collection.h"

#include "arcane/tests/StdMeshVariables.h"

#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/VariableCollection.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Traits> StdMeshVariables<Traits>::
StdMeshVariables(const MeshHandle& mesh_handle,const String& basestr,const String& base2str)
: m_byte(VariableBuildInfo(mesh_handle,basestr+base2str+"Byte"))
, m_real(VariableBuildInfo(mesh_handle,basestr+base2str+"Real"))
, m_int64(VariableBuildInfo(mesh_handle,basestr+base2str+"Int64"))
, m_int32(VariableBuildInfo(mesh_handle,basestr+base2str+"Int32"))
, m_int16(VariableBuildInfo(mesh_handle,basestr+base2str+"Int16"))
, m_int8(VariableBuildInfo(mesh_handle,basestr+base2str+"Int8"))
, m_bfloat16(VariableBuildInfo(mesh_handle,basestr+base2str+"BFloat16"))
, m_float16(VariableBuildInfo(mesh_handle,basestr+base2str+"Float16"))
, m_float32(VariableBuildInfo(mesh_handle,basestr+base2str+"Float32"))
, m_real2(VariableBuildInfo(mesh_handle,basestr+base2str+"Real2"))
, m_real2x2(VariableBuildInfo(mesh_handle,basestr+base2str+"Real2x2"))
, m_real3(VariableBuildInfo(mesh_handle,basestr+base2str+"Real3"))
, m_real3x3(VariableBuildInfo(mesh_handle,basestr+base2str+"Real3x3"))
, m_mesh_handle(mesh_handle)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Traits> StdMeshVariables<Traits>::
StdMeshVariables(const MeshHandle& mesh_handle,const String& basestr,
                 const String& base2str,const String& family_name)
: m_byte(VariableBuildInfo(mesh_handle,basestr+base2str+"Byte",family_name))
, m_real(VariableBuildInfo(mesh_handle,basestr+base2str+"Real",family_name))
, m_int64(VariableBuildInfo(mesh_handle,basestr+base2str+"Int64",family_name))
, m_int32(VariableBuildInfo(mesh_handle,basestr+base2str+"Int32",family_name))
, m_int16(VariableBuildInfo(mesh_handle,basestr+base2str+"Int16",family_name))
, m_int8(VariableBuildInfo(mesh_handle,basestr+base2str+"Int8",family_name))
, m_bfloat16(VariableBuildInfo(mesh_handle,basestr+base2str+"BFloat16",family_name))
, m_float16(VariableBuildInfo(mesh_handle,basestr+base2str+"Float16",family_name))
, m_float32(VariableBuildInfo(mesh_handle,basestr+base2str+"Float32",family_name))
, m_real2(VariableBuildInfo(mesh_handle,basestr+base2str+"Real2",family_name))
, m_real2x2(VariableBuildInfo(mesh_handle,basestr+base2str+"Real2x2",family_name))
, m_real3(VariableBuildInfo(mesh_handle,basestr+base2str+"Real3",family_name))
, m_real3x3(VariableBuildInfo(mesh_handle,basestr+base2str+"Real3x3",family_name))
, m_mesh_handle(mesh_handle)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Traits> void StdMeshVariables<Traits>::
synchronize()
{
  //NOTE: il ne faut pas appeler ces methodes sur les particules
  m_byte.synchronize();
  m_real.synchronize();
  m_int64.synchronize();
  m_int32.synchronize();
  m_int16.synchronize();
  m_int8.synchronize();
  m_bfloat16.synchronize();
  m_float16.synchronize();
  m_float32.synchronize();
  m_real2.synchronize();
  m_real2x2.synchronize();
  m_real3.synchronize();
  m_real3x3.synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Traits> void StdMeshVariables<Traits>::
addToCollection(VariableCollection vars)
{
  vars.add(m_byte.variable());
  vars.add(m_real.variable());
  vars.add(m_int64.variable());
  vars.add(m_int32.variable());
  vars.add(m_int16.variable());
  vars.add(m_int8.variable());
  vars.add(m_bfloat16.variable());
  vars.add(m_float16.variable());
  vars.add(m_float32.variable());
  vars.add(m_real2.variable());
  vars.add(m_real2x2.variable());
  vars.add(m_real3.variable());
  vars.add(m_real3x3.variable());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class StdMeshVariables< StdMeshVariableTraits2<Node,0> >;
template class StdMeshVariables< StdMeshVariableTraits2<Edge,0> >;
template class StdMeshVariables< StdMeshVariableTraits2<Face,0> >;
template class StdMeshVariables< StdMeshVariableTraits2<Cell,0> >;
template class StdMeshVariables< StdMeshVariableTraits2<Particle,0> >;

template class StdMeshVariables< StdMeshVariableTraits2<Node,1> >;
template class StdMeshVariables< StdMeshVariableTraits2<Edge,1> >;
template class StdMeshVariables< StdMeshVariableTraits2<Face,1> >;
template class StdMeshVariables< StdMeshVariableTraits2<Cell,1> >;
template class StdMeshVariables< StdMeshVariableTraits2<Particle,1> >;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
