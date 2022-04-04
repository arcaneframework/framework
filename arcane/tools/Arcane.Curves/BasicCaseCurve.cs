//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
namespace Arcane.Curves
{
  public interface ICaseCurveReader
  {
    ICurve Read();
  }
  
  public class BasicCaseCurve : ICaseCurve
  {
    string m_name;
    public string Name { get { return m_name; } }
    
    public bool IsRead { get { return m_curve!=null; } }
    
    ICurve m_curve;
    ICaseCurveReader m_reader;
    
    public BasicCaseCurve (string name,ICaseCurveReader reader)
    {
      m_name = name;
      m_reader = reader;
    }
    
    public BasicCaseCurve (ICurve curve)
    {
      m_curve = curve;
      m_name = curve.Name;
    }

    public ICurve Read()
    {
      //TODO ajoute verrou pour multi-thread ?
      if (m_curve!=null)
        return m_curve;
      m_curve = m_reader.Read();
      // TODO: faire 'm_reader==null' pour permettre au garbage collector
      // de libérer m_reader et donc l'éventuelle mémoire associée.
      return m_curve;
    }
  }
}

