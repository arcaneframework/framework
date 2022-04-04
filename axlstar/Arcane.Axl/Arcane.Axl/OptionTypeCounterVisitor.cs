//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.IO;

namespace Arcane.Axl
{
  /**
   * Classe implémentant le Design Pattern du visiteur pour compter le nombre
   * d'options de chaque type.
   */
  public class OptionTypeCounterVisitor : IOptionInfoVisitor
  {
    public OptionTypeCounterVisitor()
    {
      m_predicate = o => true;
    }

    public OptionTypeCounterVisitor(Func<Option, bool> predicate)
    {
      m_predicate = predicate;
    }

    public virtual void VisitComplex(ComplexOptionInfo info)
    {
      ++m_nb_complex;
      info.AcceptChildren(this, m_predicate);
    }

    public virtual void VisitExtended(ExtendedOptionInfo info) { ++m_nb_extended; }
    public virtual void VisitEnumeration(EnumerationOptionInfo info) { ++m_nb_enumeration; }
    public virtual void VisitScript(ScriptOptionInfo info) { ++m_nb_script; }
    public virtual void VisitSimple(SimpleOptionInfo info) { ++m_nb_simple; }
    public virtual void VisitServiceInstance(ServiceInstanceOptionInfo info) { ++m_nb_service_instance; }

    public int NbComplex { get { return m_nb_complex; } }
    public int NbExtended { get { return m_nb_extended; } }
    public int NbEnumeration { get { return m_nb_enumeration; } }
    public int NbScript { get { return m_nb_script; } }
    public int NbSimple { get { return m_nb_simple; } }
    public int NbServiceInstance { get { return m_nb_service_instance; } }

    public int NbTotalOption
    {
      get
      {
        return m_nb_complex + m_nb_extended + m_nb_enumeration +
        m_nb_script + m_nb_simple + m_nb_service_instance;
      }
    }
    private int m_nb_complex;
    private int m_nb_extended;
    private int m_nb_enumeration;
    private int m_nb_script;
    private int m_nb_simple;
    private int m_nb_service_instance;
    private Func<Option, bool> m_predicate;
  }
}
