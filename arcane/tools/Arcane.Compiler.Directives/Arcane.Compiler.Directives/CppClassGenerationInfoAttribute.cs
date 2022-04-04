//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

namespace Arcane.Compiler.Directives
{
  /// <summary>
  /// Permet de specifier les infos pour cette classe lors de la generation en C++
  /// </summary>
  public class CppClassGenerationInfoAttribute : Attribute
  {
    private string m_full_name;
    /// <value>
    /// Nom C++ complet de la classe generee
    /// </value>
    public string FullName { get { return m_full_name; } set { m_full_name = value; } }
    
    private TypeMapping m_mapping;
    /// <value>
    /// Indique le mapping de type de la classe
    /// </value>
    public TypeMapping TypeMapping { get { return m_mapping; } set { m_mapping = value; } }
    
    private bool m_not_generated;
    /// <value>
    /// Indique si la classe ne doit etre generee en C++
    /// </value>
    public bool NotGenerated { get { return m_not_generated; } set { m_not_generated = value; } }
    
    public CppClassGenerationInfoAttribute()
    {
    }
  }
}
