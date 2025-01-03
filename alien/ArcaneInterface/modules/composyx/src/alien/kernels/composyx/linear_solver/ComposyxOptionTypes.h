// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once
struct ComposyxOptionTypes
{
  enum eSolver
  {
    BiCGStab,
    CG,
    GMRES
  };

  enum ePreconditioner
  {
    None,
    DiagPC,
    ASPC
  };
};

