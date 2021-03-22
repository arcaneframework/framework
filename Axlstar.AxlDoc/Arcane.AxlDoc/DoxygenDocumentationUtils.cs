//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Génération de la documentation au format Doxygen.                         */
/*---------------------------------------------------------------------------*/

using System;
using Arcane.Axl;

namespace Arcane.AxlDoc
{
  public static class DoxygenDocumentationUtils
  {
    static public string AnchorName (Option option)
    {
      if (option == null)
        throw new ArgumentException ("null option");

      ServiceOrModuleInfo base_info = option.ServiceOrModule;
      if (base_info == null)
        throw new ArgumentException (String.Format ("null 'ServiceOrModuleInfo' for option '{0}'", option.FullName));

      if (option.Name == null)
        throw new ArgumentException (String.Format ("option '{0}' has no name (type={1})", option.NodeName, option.Type));
      string aname = "axldoc_" + option.ServiceOrModule.Name + "_" + option.FullName.Replace ('/', '_').Replace ('-', '_');
      return aname;
    }

    static public string AnchorName(String intface)
    {
      if (intface == null)
        throw new ArgumentException("null option");
      return intface.Replace(':','_');
    }
  }

}
