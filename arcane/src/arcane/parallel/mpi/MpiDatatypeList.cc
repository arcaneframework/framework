// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiDatatypeList.cc                                          (C) 2000-2020 */
/*                                                                           */
/* Gestionnaire de parallélisme utilisant MPI.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/HPReal.h"
#include "arcane/utils/APReal.h"

#include "arcane/parallel/mpi/MpiDatatypeList.h"
#include "arcane/parallel/mpi/MpiDatatype.h"

#include "arcane/SerializeBuffer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiDatatypeList::
MpiDatatypeList(bool is_ordered_reduce)
: m_is_ordered_reduce(is_ordered_reduce)
, m_char(nullptr)
, m_unsigned_char(nullptr)
, m_signed_char(nullptr)
, m_short(nullptr)
, m_unsigned_short(nullptr)
, m_int(nullptr)
, m_unsigned_int(nullptr)
, m_long(nullptr)
, m_unsigned_long(nullptr)
, m_long_long(nullptr)
, m_unsigned_long_long(nullptr)
, m_float(nullptr)
, m_double(nullptr)
, m_long_double(nullptr)
, m_apreal(nullptr)
, m_real2(nullptr)
, m_real3(nullptr)
, m_real2x2(nullptr)
, m_real3x3(nullptr)
, m_hpreal(nullptr)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiDatatypeList::
~MpiDatatypeList()
{
  delete m_char;
  delete m_signed_char;
  delete m_unsigned_char;
  delete m_short;
  delete m_unsigned_short;
  delete m_int;
  delete m_unsigned_int;
  delete m_long;
  delete m_unsigned_long;
  delete m_long_long;
  delete m_unsigned_long_long;
  delete m_apreal;
  delete m_float;
  delete m_double;
  delete m_long_double;
  delete m_real2;
  delete m_real3;
  delete m_real2x2;
  delete m_real3x3;
  delete m_hpreal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiDatatype* MpiDatatypeList::
datatype(char)
{
  return m_char;
}
MpiDatatype* MpiDatatypeList::
datatype(unsigned char)
{
  return m_unsigned_char;
}
MpiDatatype* MpiDatatypeList::
datatype(signed char)
{
  return m_signed_char;
}
MpiDatatype* MpiDatatypeList::
datatype(short)
{
  return m_short;
}
MpiDatatype* MpiDatatypeList::
datatype(int)
{
  return m_int;
}
MpiDatatype* MpiDatatypeList::
datatype(float)
{
  return m_float;
}
MpiDatatype* MpiDatatypeList::
datatype(double)
{
  return m_double;
}
MpiDatatype* MpiDatatypeList::
datatype(long double)
{
  return m_long_double;
}

MpiDatatype* MpiDatatypeList::
datatype(long int)
{
  return m_long;
}
MpiDatatype* MpiDatatypeList::
datatype(unsigned short)
{
  return m_unsigned_short;
}

MpiDatatype* MpiDatatypeList::
datatype(unsigned int)
{
  return m_unsigned_int;
}

MpiDatatype* MpiDatatypeList::
datatype(unsigned long)
{
  return m_unsigned_long;
}

MpiDatatype* MpiDatatypeList::
datatype(long long)
{
  return m_long_long;
}
MpiDatatype* MpiDatatypeList::
datatype(unsigned long long)
{
  return m_unsigned_long_long;
}

MpiDatatype* MpiDatatypeList::
datatype(Real2)
{
  return m_real2;
}
MpiDatatype* MpiDatatypeList::
datatype(Real3)
{
  return m_real3;
}
MpiDatatype* MpiDatatypeList::
datatype(Real2x2)
{
  return m_real2x2;
}
MpiDatatype* MpiDatatypeList::
datatype(Real3x3)
{
  return m_real3x3;
}
MpiDatatype* MpiDatatypeList::
datatype(HPReal)
{
  return m_hpreal;
}
MpiDatatype* MpiDatatypeList::
datatype(APReal)
{
  return m_apreal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDatatypeList::
_init()
{
  bool is_commutative = (m_is_ordered_reduce) ? 0 : 1;
    
  MPI_Datatype real_mpi_datatype = MpiBuiltIn::datatype(Real());

  m_char = new MpiDatatype(MPI_CHAR);
  m_signed_char = new MpiDatatype(MPI_CHAR);
  m_unsigned_char = new MpiDatatype(MPI_CHAR);

  m_short = new MpiDatatype(MPI_SHORT);
  m_unsigned_short = new MpiDatatype(MPI_UNSIGNED_SHORT);

  m_int = new MpiDatatype(MPI_INT);
  m_unsigned_int= new MpiDatatype(MPI_UNSIGNED);

  m_long = new MpiDatatype(MPI_LONG);
  m_unsigned_long = new MpiDatatype(MPI_UNSIGNED_LONG);

  m_long_long = new MpiDatatype(MPI_LONG_LONG);
  m_unsigned_long_long = new MpiDatatype(MPI_UNSIGNED_LONG_LONG);

  // Si on veut une reduction ordonnée, il faut redéfinir l'opérateur de réduction
  // pour les flottants
  if (is_commutative){
    m_double = new MpiDatatype(MPI_DOUBLE,true,new StdMpiReduceOperator<double>(is_commutative));
    m_long_double = new MpiDatatype(MPI_LONG_DOUBLE,true,new StdMpiReduceOperator<long double>(is_commutative));
    m_float = new MpiDatatype(MPI_FLOAT,true,new StdMpiReduceOperator<float>(is_commutative));
  }
  else{
    m_double = new MpiDatatype(MPI_DOUBLE);
    m_long_double = new MpiDatatype(MPI_LONG_DOUBLE);
    m_float = new MpiDatatype(MPI_FLOAT);
  }
  {
    // Real2
    MPI_Datatype mpi_datatype;
    MPI_Type_contiguous(2,real_mpi_datatype,&mpi_datatype);
    MPI_Type_commit(&mpi_datatype);
    m_real2 = new MpiDatatype(mpi_datatype,false,new StdMpiReduceOperator<Real2>(is_commutative));
  }
  {
    // Real3
    MPI_Datatype mpi_datatype;
    MPI_Type_contiguous(3,real_mpi_datatype,&mpi_datatype);
    MPI_Type_commit(&mpi_datatype);
    m_real3 = new MpiDatatype(mpi_datatype,false,new StdMpiReduceOperator<Real3>(is_commutative));
  }
  {
    // Real2x2
    MPI_Datatype mpi_datatype;
    MPI_Type_contiguous(4,real_mpi_datatype,&mpi_datatype);
    MPI_Type_commit(&mpi_datatype);
    m_real2x2 = new MpiDatatype(mpi_datatype,false,new StdMpiReduceOperator<Real2x2>(is_commutative));
  }
  {
    // Real3x3
    MPI_Datatype mpi_datatype;
    MPI_Type_contiguous(9,real_mpi_datatype,&mpi_datatype);
    MPI_Type_commit(&mpi_datatype);
    m_real3x3 = new MpiDatatype(mpi_datatype,false,new StdMpiReduceOperator<Real3x3>(is_commutative));
  }
  {
    // HPReal
    MPI_Datatype mpi_datatype;
    MPI_Type_contiguous(2,real_mpi_datatype,&mpi_datatype);
    MPI_Type_commit(&mpi_datatype);
    m_hpreal = new MpiDatatype(mpi_datatype,false,new StdMpiReduceOperator<HPReal>(is_commutative));
  }
  {
    // APReal
    MPI_Datatype mpi_datatype;
    MPI_Type_contiguous(4,real_mpi_datatype,&mpi_datatype);
    MPI_Type_commit(&mpi_datatype);
    m_apreal = new MpiDatatype(mpi_datatype,false,new StdMpiReduceOperator<APReal>(is_commutative));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
