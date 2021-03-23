// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
//#include <arcane/matrix/Matrix.h>
//#include <arcane/matrix/MatrixOperationsT.h>

#include "Matrix.h"
#include "MatrixOperationsT.h"

#include "Vector.h"
#include "VectorOperationsT.h"

ARCANE_BEGIN_NAMESPACE

Vector vector_test()
{
  IndexedSpace i;
  IndexedSpace j;

  Vector x(i), y(j);

  Vector z = 3.14*x-y;

  return z;
}

Matrix matrix_test() 
{
  Matrix a,b;
  Matrix c = a + 2.05*b;

  return c;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int main(int argc, char** argv)
{
  arcane::vector_test();
  //arcane::matrix_test();
  return 0;
}

