/*---------------------------------------------------------------------------*/
/* DoxygenDocumentationUtils.cc                                (C) 2000-2007 */
/*                                                                           */
/* Génération de la documentation au format Doxygen.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace Arcane.Axl
{
  public static class DoxygenDocumentationUtils
  {
    static public string AnchorName(Option option)
    {
      if (option == null)
        throw new ArgumentException("null option");

      ServiceOrModuleInfo base_info = option.ServiceOrModule;
      if (base_info == null)
        throw new ArgumentException(String.Format("null 'ServiceOrModuleInfo' for option '{0}'", option.FullName));

      if (option.Name == null)
        throw new ArgumentException(String.Format("option '{0}' has no name (type={1})", option.NodeName, option.Type));
      string aname = "axldoc_" + option.ServiceOrModule.Name + "_" + option.FullName.Replace('/', '_').Replace('-', '_');
      return aname;
    }
  }
}
