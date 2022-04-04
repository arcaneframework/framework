//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;

namespace Arcane.Axl.Xsd
{
  public partial class Option
  {
    public static int UNBOUNDED = -1;

    public bool IsOptional {
      get {
        return optionalSpecified ? (optional ? true : false) : false;
      }
    }

    public bool AllowsNull {
      get {
        return allownullSpecified ? (allownull ? true : false) : false;
      }
    }

    public string Default {
      get {
        return @default == null ? "Arcane::String()" : "\"" + @default + "\"";
      }
    }

    public int MinOccurs {
      get {
        if (minOccurs == null)
          return 1;
        else {
          int occurs = 0;
          bool is_ok = int.TryParse (minOccurs, out occurs);
          if (!is_ok) {
            Console.WriteLine ("Valeur invalide pour l'attribut \"minOccurs\" de l'element \""
            + name + "\". Utilise 1. (valeur='{0}')", minOccurs);
            return 1;
          } else {
            return occurs;
          }
        }
      }
    }

    public int MaxOccurs {
      get {
        if (maxOccurs == null)
          return 1;
        else if (maxOccurs == "unbounded")
          return UNBOUNDED;
        else {
          int occurs = 0;
          bool is_ok = int.TryParse(maxOccurs, out occurs);
          if (!is_ok) {
            Console.WriteLine ("Valeur invalide pour l'attribut \"maxOccurs\" de l'element \""
            + name + "\". Utilise 1. (valeur='{0}')", maxOccurs);
            return 1;
          } else {
            return occurs;
          }
        }
      }
    }

    public bool IsMulti { get { return MinOccurs != 1 || MaxOccurs != 1; } }

    public bool IsSingle { get { return IsMulti == false; } }
  }
}

