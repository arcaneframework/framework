//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.IO;

namespace Arcane.Curves
{

/// <summary>
/// Liste des courbes d'un cas.
/// </summary>
  public class CaseCurves : IDisposable
  {
    Dictionary<string,ICaseCurve> m_curves;
    List<ICaseCurve> m_curves_interfaces;

    //TODO: utiliser un enumerator
    public IList<ICaseCurve> Curves { get { return m_curves_interfaces; } }

    /// <summary>
    /// Retourne la courbe de nom \a name. Lance une exception si la courbe n'est pas trouvee.
    /// </summary>
    /// <param name="name">
    /// A <see cref="System.String"/>
    /// </param>
    /// <returns>
    /// A <see cref="ICurve"/>
    /// </returns>
    public ICaseCurve GetCurve(string name)
    {
      return m_curves[name];
    }
    
    public bool TryGetCurve(string name,out ICaseCurve curve)
    {
      return m_curves.TryGetValue(name,out curve);
    }
    
    public CaseCurves ()
    {
      m_curves = new Dictionary<string, ICaseCurve>();
      m_curves_interfaces = new List<ICaseCurve>();
    }

    public void AddCurve(ICaseCurve curve)
    {
      m_curves.Add(curve.Name,curve);
      m_curves_interfaces.Add(curve);
    }
    
    public virtual void Dispose()
    {
    }

    /// <summary>
    /// Lit les courbes d'un cas. \a curve_path est le repertoire courbe de ce cas.
    /// Il doit contenir soit un fichier 'curves.acv', soit un fichier 'time_history.xml'
    /// et un repertoire 'gnuplot'
    /// </summary>
    /// <param name="curve_path">
    /// A <see cref="System.String"/>
    /// </param>
    /// <returns>
    /// A <see cref="CaseCurves"/>
    /// </returns>
    public static CaseCurves ReadCase(string curve_path)
    {
      if (!Directory.Exists(curve_path))
        throw new ArgumentException(String.Format("argument '{0}' is not a directory",curve_path));
      string file = Path.Combine(curve_path,"curves.acv");
      if (File.Exists(file)){
        Console.WriteLine("Using 'curves.acv' file path={0} machine={1}",curve_path,Environment.MachineName);
        ArcaneCaseReader reader = ArcaneCaseReader.CreateFromFile(file);
        return reader.CaseCurves;
      }
      file = Path.Combine(curve_path,"time_history.xml");
      if (File.Exists(file)){
        Console.WriteLine("Using 'gnuplot' files");
        GnuplotCaseReader greader = new GnuplotCaseReader();
        greader.ReadPath(curve_path);
        return greader.CaseCurves;
      }
      throw new ApplicationException(String.Format("Curve path must contain 'curves.acv' or 'time_history.xml' : path={0} machine={1}",
                                                   curve_path,Environment.MachineName));
    }
  }
}
