// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

namespace Alien {

class ILinearSystem;
class LocalLinearSystem;

class ILinearSystemVisitor
{

public:
  virtual ~ILinearSystemVisitor() { }

  /**
   *  Initialise
   */
  virtual void init() = 0;

  virtual bool visit(      ILinearSystem * system) { throw Arccore::FatalErrorException(A_FUNCINFO,"not implemented");  }
  virtual bool visit(  LocalLinearSystem * system) { throw Arccore::FatalErrorException(A_FUNCINFO,"not implemented");  }
};
}

