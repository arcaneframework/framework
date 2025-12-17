// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "HypreInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien::Internal {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixInternal::~MatrixInternal()
{
  if (m_internal)
    HYPRE_IJMatrixDestroy(m_internal);
}

VectorInternal::~VectorInternal()
{
  if (m_internal)
    HYPRE_IJVectorDestroy(m_internal);
}

/*---------------------------------------------------------------------------*/

bool
MatrixInternal::init(const HYPRE_Int ilower, const HYPRE_Int iupper, const HYPRE_Int jlower,
    const int jupper, const Arccore::ConstArrayView<Arccore::Integer>& lineSizes)
{
  HYPRE_ClearAllErrors() ;

  int ierr = 0; // code d'erreur de retour

  // -- Matrix --
  ierr = HYPRE_IJMatrixCreate(m_comm, ilower, iupper, jlower, jupper, &m_internal);
  ierr |= HYPRE_IJMatrixSetObjectType(m_internal, HYPRE_PARCSR);
  if(lineSizes.size()>0)
     ierr |= HYPRE_IJMatrixSetRowSizes(m_internal, lineSizes.unguardedBasePointer());
  ierr |= HYPRE_IJMatrixInitialize(m_internal);
  return (ierr == 0);
}

bool
VectorInternal::init(const int ilower, const int iupper)
{
  // -- B Vector --
  int ierr = HYPRE_IJVectorCreate(m_comm, ilower, iupper, &m_internal);
  ierr |= HYPRE_IJVectorSetObjectType(m_internal, HYPRE_PARCSR);
  ierr |= HYPRE_IJVectorInitialize(m_internal);

  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
MatrixInternal::addMatrixValues(const int nrow, const int* rows, const int* ncols,
    const int* cols, const Arccore::Real* values)
{
  int ierr = HYPRE_IJMatrixAddToValues(
      m_internal, nrow, const_cast<int*>(ncols), rows, cols, values);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
MatrixInternal::setMatrixValues(const int nrow, const int* rows, const int* ncols,
    const int* cols, const Arccore::Real* values)
{
  int ierr = HYPRE_IJMatrixSetValues(
      m_internal, nrow, const_cast<int*>(ncols), rows, cols, values);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
VectorInternal::addValues(const int nrow, const int* rows, const Arccore::Real* values)
{
  int ierr = HYPRE_IJVectorAddToValues(m_internal,
      nrow, // nb de valeurs
      rows, values);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
VectorInternal::setValues(const int nrow, const int* rows, const Arccore::Real* values)
{
  int ierr = HYPRE_IJVectorSetValues(m_internal,
      nrow, // nb de valeurs
      rows, values);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
VectorInternal::setInitValues(
    const int nrow, const int* rows, const Arccore::Real* values)
{
  int ierr = HYPRE_IJVectorSetValues(m_internal,
      nrow, // nb de valeurs
      rows, values);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
MatrixInternal::assemble()
{
  int ierr = HYPRE_IJMatrixAssemble(m_internal);
  return (ierr == 0);
}

bool
VectorInternal::assemble()
{
  int ierr = HYPRE_IJVectorAssemble(m_internal);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
VectorInternal::getValues(const int nrow, const int* rows, Arccore::Real* values)
{
  int ierr;
  ierr = HYPRE_IJVectorGetValues(m_internal, nrow, rows, values);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
