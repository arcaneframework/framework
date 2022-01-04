// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiDatatyes.h                                               (C) 2000-2020 */
/*                                                                           */
/* Gère les MPI_Datatype associées aux types Arcane.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPIDATATYPELIST_H
#define ARCANE_PARALLEL_MPI_MPIDATATYPELIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/parallel/mpi/ArcaneMpi.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère les MPI_Datatype associées aux types Arcane.
 */
class ARCANE_MPI_EXPORT MpiDatatypeList
{
 public:

  MpiDatatypeList(bool is_ordered_reduce);
  ~MpiDatatypeList();

 public:

  MpiDatatype* datatype(char);
  MpiDatatype* datatype(signed char);
  MpiDatatype* datatype(unsigned char);
  MpiDatatype* datatype(short);
  MpiDatatype* datatype(int);
  MpiDatatype* datatype(float);
  MpiDatatype* datatype(double);
  MpiDatatype* datatype(long double);
  MpiDatatype* datatype(long);
  MpiDatatype* datatype(unsigned short);
  MpiDatatype* datatype(unsigned int);
  MpiDatatype* datatype(unsigned long);
  MpiDatatype* datatype(unsigned long long);
  MpiDatatype* datatype(long long);

  MpiDatatype* datatype(APReal);
  MpiDatatype* datatype(Real2);
  MpiDatatype* datatype(Real3);
  MpiDatatype* datatype(Real2x2);
  MpiDatatype* datatype(Real3x3);
  MpiDatatype* datatype(HPReal);

 private:
  
  bool m_is_ordered_reduce;
  MpiDatatype* m_char;
  MpiDatatype* m_unsigned_char;
  MpiDatatype* m_signed_char;
  MpiDatatype* m_short;
  MpiDatatype* m_unsigned_short;
  MpiDatatype* m_int;
  MpiDatatype* m_unsigned_int;
  MpiDatatype* m_long;
  MpiDatatype* m_unsigned_long;
  MpiDatatype* m_long_long;
  MpiDatatype* m_unsigned_long_long;
  MpiDatatype* m_float;
  MpiDatatype* m_double;
  MpiDatatype* m_long_double;
  MpiDatatype* m_apreal;
  MpiDatatype* m_real2;
  MpiDatatype* m_real3;
  MpiDatatype* m_real2x2;
  MpiDatatype* m_real3x3;
  MpiDatatype* m_hpreal;

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
