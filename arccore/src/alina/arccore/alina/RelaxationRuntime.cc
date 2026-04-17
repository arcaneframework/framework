// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RelaxationRuntime.cc                                        (C) 2000-2026 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/RelaxationRuntime.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream& operator<<(std::ostream& os, eRelaxationType r)
{
  switch (r) {
  case eRelaxationType::gauss_seidel:
    return os << "gauss_seidel";
  case eRelaxationType::ilu0:
    return os << "ilu0";
  case eRelaxationType::iluk:
    return os << "iluk";
  case eRelaxationType::ilup:
    return os << "ilup";
  case eRelaxationType::ilut:
    return os << "ilut";
  case eRelaxationType::damped_jacobi:
    return os << "damped_jacobi";
  case eRelaxationType::spai0:
    return os << "spai0";
  case eRelaxationType::spai1:
    return os << "spai1";
  case eRelaxationType::chebyshev:
    return os << "chebyshev";
  default:
    return os << "???";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::istream& operator>>(std::istream& in, eRelaxationType& r)
{
  std::string val;
  in >> val;

  if (val == "gauss_seidel")
    r = eRelaxationType::gauss_seidel;
  else if (val == "ilu0")
    r = eRelaxationType::ilu0;
  else if (val == "iluk")
    r = eRelaxationType::iluk;
  else if (val == "ilup")
    r = eRelaxationType::ilup;
  else if (val == "ilut")
    r = eRelaxationType::ilut;
  else if (val == "damped_jacobi")
    r = eRelaxationType::damped_jacobi;
  else if (val == "spai0")
    r = eRelaxationType::spai0;
  else if (val == "spai1")
    r = eRelaxationType::spai1;
  else if (val == "chebyshev")
    r = eRelaxationType::chebyshev;
  else
    throw std::invalid_argument("Invalid relaxation value. Valid choices are:"
                                "gauss_seidel, ilu0, iluk, ilup, ilut, damped_jacobi, spai0, spai1, chebyshev.");

  return in;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
