// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DistributedSolver.h                                         (C) 2000-2026 */
/*                                                                           */
/* Adapters to handle distribution other standard solvers.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_MPI_DISTRIBUTEDSOLVER_H
#define ARCCORE_ALINA_MPI_DISTRIBUTEDSOLVER_H
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

#include "arccore/alina/ConjugateGradientSolver.h"
#include "arccore/alina/BiCGStabSolver.h"
#include "arccore/alina/BiCGStabLSolver.h"
#include "arccore/alina/FlexibleGMRESSolver.h"
#include "arccore/alina/GMRESSolver.h"
#include "arccore/alina/IDRSSolver.h"
#include "arccore/alina/LooseGMRESSolver.h"
#include "arccore/alina/PreconditionerOnlySolver.h"
#include "arccore/alina/RichardsonSolver.h"
#include "arccore/alina/DistributedInnerProduct.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend, class InnerProduct = DistributedInnerProduct>
class DistributedConjugateGradientSolver
: public ConjugateGradientSolver<Backend, InnerProduct>
{
  typedef ConjugateGradientSolver<Backend, InnerProduct> Base;

 public:

  using Base::Base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend, class InnerProduct = DistributedInnerProduct>
class DistributedBiCGStabSolver
: public BiCGStabSolver<Backend, InnerProduct>
{
  typedef BiCGStabSolver<Backend, InnerProduct> Base;

 public:

  using Base::Base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend, class InnerProduct = DistributedInnerProduct>
class DistributedBiCGStabLSolver
: public BiCGStabLSolver<Backend, InnerProduct>
{
  typedef BiCGStabLSolver<Backend, InnerProduct> Base;

 public:

  using Base::Base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend, class InnerProduct = DistributedInnerProduct>
class DistributedFlexibleGMRESSolver
: public FlexibleGMRESSolver<Backend, InnerProduct>
{
  typedef FlexibleGMRESSolver<Backend, InnerProduct> Base;

 public:

  using Base::Base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend, class InnerProduct = DistributedInnerProduct>
class DistributedGMRESSolver
: public GMRESSolver<Backend, InnerProduct>
{
  typedef GMRESSolver<Backend, InnerProduct> Base;

 public:

  using Base::Base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend, class InnerProduct = DistributedInnerProduct>
class DistributedIDRSSolver
: public IDRSSolver<Backend, InnerProduct>
{
  typedef IDRSSolver<Backend, InnerProduct> Base;

 public:

  using Base::Base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend, class InnerProduct = DistributedInnerProduct>
class DistributedLooseGMRESSolver
: public LooseGMRESSolver<Backend, InnerProduct>
{
  typedef LooseGMRESSolver<Backend, InnerProduct> Base;

 public:

  using Base::Base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend, class InnerProduct = DistributedInnerProduct>
class DistributedPreconditionerOnlySolver
: public PreconditionerOnlySolver<Backend, InnerProduct>
{
  typedef PreconditionerOnlySolver<Backend, InnerProduct> Base;

 public:

  using Base::Base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend, class InnerProduct = DistributedInnerProduct>
class RichardsonSolve
: public RichardsonSolver<Backend, InnerProduct>
{
  typedef RichardsonSolver<Backend, InnerProduct> Base;

 public:

  using Base::Base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
