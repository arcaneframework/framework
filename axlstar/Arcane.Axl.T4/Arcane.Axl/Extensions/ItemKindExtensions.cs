//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;

namespace Arcane.Axl
{
  public static class ItemKindExtensions
  {
    public static string Name(this Xsd.ItemKind type)
    {
      switch(type){
      case Xsd.ItemKind.cell:
        return "Cell";
      case Xsd.ItemKind.edge:
        return "Edge";
      case Xsd.ItemKind.face:
        return "Face";
      case Xsd.ItemKind.node:
        return "Node";
      case Xsd.ItemKind.particle:
        return "Particle";
      case Xsd.ItemKind.dof:
        return "DoF";
      default:
        throw new TypeUnloadedException (); 
      }
    }
  }
}

