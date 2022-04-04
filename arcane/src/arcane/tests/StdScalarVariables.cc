// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdScalarVariables.cc                                       (C) 2000-2019 */
/*                                                                           */
/* Définition de variables scalaires pour les tests.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/IMesh.h"

#include "arcane/tests/StdScalarVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StdScalarVariables::
StdScalarVariables(const MeshHandle& mesh_handle,const String& basestr)
: m_byte(VariableBuildInfo(mesh_handle,basestr+"Byte"))
, m_real(VariableBuildInfo(mesh_handle,basestr+"Real"))
, m_int64(VariableBuildInfo(mesh_handle,basestr+"Int64"))
, m_int32(VariableBuildInfo(mesh_handle,basestr+"Int32"))
, m_int16(VariableBuildInfo(mesh_handle,basestr+"Int16"))
, m_real2(VariableBuildInfo(mesh_handle,basestr+"Real2"))
, m_real2x2(VariableBuildInfo(mesh_handle,basestr+"Real2x2"))
, m_real3(VariableBuildInfo(mesh_handle,basestr+"Real3"))
, m_real3x3(VariableBuildInfo(mesh_handle,basestr+"Real3x3"))
, m_mesh_handle(mesh_handle)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StdScalarVariables::
setValues(Integer iteration)
{
  Int64 x1 = 5;
  Int64 x2 = iteration;
  Int64 n = 1 + x1 * (x2 * x2);
  Real r = Convert::toReal(n);

  this->m_byte = (Byte)(n % 255);
  this->m_real = r;
  this->m_int64 = n;
  this->m_int32 = static_cast<Int32>(n+1);
  this->m_int16 = static_cast<Int16>(n+2);
  this->m_real2 = Real2 (r, r+1);
  this->m_real2x2 = Real2x2::fromLines (r, r+1., r+2., r+3.);
  this->m_real3 = Real3 (r, r+1., r+2.0);
  this->m_real3x3 = Real3x3::fromLines (r, r+1., r+2., r+3., r+4., r+5., r+6., r+7., r+8.);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> void
_writeError(ITraceMng* tm,const char* type_name,const DataType& value,
            const DataType& expected_value)
{
  tm->info() << "Bad scalar value type=" << type_name
             << " value=" << value
             << " expected=" << expected_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define CHECK_VALUE(type_name)\
    if (type_name##_ref!=type_name##_current){\
      ++nb_error;\
      if (nb_error<nb_display_error){\
        _writeError(tm,#type_name,type_name##_current,type_name##_ref);\
      }\
    }

Integer StdScalarVariables::
checkValues(Integer iteration)
{
  Integer nb_error = 0;
  ITraceMng* tm = this->m_mesh_handle.mesh()->traceMng();
  Integer nb_display_error = 10;

  Int64 x1 = 5;
  Int64 x2 = iteration;
  Int64 n = 1 + (x1 * x2 * x2);
  Real r = Convert::toReal(n);
  Int32 i32 = (Int32)(n+1);
  Int16 i16 = (Int16)(n+2);

  Byte byte_current = this->m_byte();
  Byte byte_ref = (Byte)(n % 255);

  Real real_current = this->m_real();
  Real real_ref = r;

  Int64 int64_current = this->m_int64();
  Int64 int64_ref = n;

  Int32 int32_current = this->m_int32();
  Int32 int32_ref = i32;

  Int32 int16_current = this->m_int16();
  Int32 int16_ref = i16;

  Real2 real2_current = this->m_real2();
  Real2 real2_ref = Real2(r,r+1.0);

  Real2x2 real2x2_current = this->m_real2x2();
  Real2x2 real2x2_ref = Real2x2::fromLines(r,r+1.0,r+2.0,r+3.0);

  Real3 real3_current = this->m_real3();
  Real3 real3_ref = Real3(r,r+1.0,r+2.0);

  Real3x3 real3x3_current = this->m_real3x3();
  Real3x3 real3x3_ref = Real3x3::fromLines(r,r+1.0,r+2.0,r+3.0,r+4.0,r+5.0,r+6.0,r+7.0,r+8.0);

  CHECK_VALUE(byte);
  CHECK_VALUE(real);
  CHECK_VALUE(int64);
  CHECK_VALUE(int32);
  CHECK_VALUE(int16);
  CHECK_VALUE(real2);
  CHECK_VALUE(real2x2);
  CHECK_VALUE(real3);
  CHECK_VALUE(real3x3);

  return nb_error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer StdScalarVariables::
checkReplica()
{
  Integer nb_error = 0;

  Integer max_print = 5;
  nb_error += this->m_byte.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_real.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_int64.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_int32.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_int16.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_real2.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_real3.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_real2x2.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_real3x3.checkIfSameOnAllReplica(max_print);

  return nb_error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
