// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ScopedStreamModifier.h                                      (C) 2000-2026 */
/*                                                                           */
/* Save/Restore std::stream flags.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_SCOPEDSTREAMMODIFIER_H
#define ARCCORE_ALINA_SCOPEDSTREAMMODIFIER_H
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

#include <ios>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Save ostream flags in constructor, restore in destructor.
 */
class ScopedStreamModifier
{
 public:

  ScopedStreamModifier(std::ios_base& s)
  : s(s)
  , f(s.flags())
  , p(s.precision())
  {}
  ~ScopedStreamModifier()
  {
    s.flags(f);
    s.precision(p);
  }

 private:

  std::ios_base& s;
  std::ios_base::fmtflags f;
  std::streamsize p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
