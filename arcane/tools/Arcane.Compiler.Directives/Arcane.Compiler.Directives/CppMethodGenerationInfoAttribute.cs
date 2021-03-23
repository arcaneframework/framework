//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

namespace Arcane.Compiler.Directives
{
  /// <summary>
  /// Permet de specifier les infos pour cette methode lors de la generation en C++
  /// </summary>
  public class CppMethodGenerationInfoAttribute : Attribute
  {
    private string m_name;
    /// <value>
    /// Nom C++ de la methode generee
    /// </value>
    public string Name { get { return m_name; } set { m_name = value; } }
    
    private bool m_is_inline;
    /// <value>
    /// Indique si le methode doit etre inline.
    /// </value>
    public bool IsInline { get { return m_is_inline; } set { m_is_inline = true; } }
    
    private bool m_is_field;
    /// <value>
    /// Indique si la methode est mappee en C++ comme un champ de classe.
    /// Il doit obligatoirement s'agir d'une methode sans arguments
    /// </value>
    public bool IsField { get { return m_is_field; } set { m_is_field = true; } }
    
    private bool m_not_generated;
    /// <value>
    /// Indique si la methode doit etre generee en C++
    /// </value>
    public bool NotGenerated { get { return m_not_generated; } set { m_not_generated = value; } }

    public CppMethodGenerationInfoAttribute()
    {
    }
  }
}
