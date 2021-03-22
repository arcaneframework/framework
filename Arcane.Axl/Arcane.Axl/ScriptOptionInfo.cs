//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ScriptOptionInfo.cs                                         (C) 2000-2007 */
/*                                                                           */
/* Classe stockant les informations de l'élément XML "script".               */
/*---------------------------------------------------------------------------*/
using System.Xml;

namespace Arcane.Axl
{
  /**
   * Classe stockant les informations de l'élément XML "script". 
   */
  public class ScriptOptionInfo
    : Option
  {
    public ScriptOptionInfo(OptionBuildInfo build_info)
    : base(build_info)
    {
      XmlElement node = build_info.Element;
      if (Name == null)
        AttrError(node, "name");
    }

    public override void Accept(IOptionInfoVisitor v)
    {
      v.VisitScript(this);
    }
  }
}
