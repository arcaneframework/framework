// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

struct PETScOptionTypes
{
  enum eMemoryType
  {
    HostMemory,
    DeviceMemory,
    ShareMemory
  };

  enum eExecSpace
  {
    Host,
    Device
  };
};

