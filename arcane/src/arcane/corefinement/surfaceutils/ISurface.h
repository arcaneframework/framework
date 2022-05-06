// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISurface.h                                                  (C) 2000-2022 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ISURFACE_H
#define ISURFACE_H

#include <arcane/utils/ArcaneGlobal.h>

ARCANE_BEGIN_NAMESPACE
NUMERICS_BEGIN_NAMESPACE

//! Purely virtual interface for surface representation
/*! Use as a pretty 'void*' pointer. Each implementation has to cast this 
 *  object before using it
 */
class ISurface
{
public:
  virtual ~ISurface() {}
};

NUMERICS_END_NAMESPACE
ARCANE_END_NAMESPACE

#endif /* ISURFACE_H */
