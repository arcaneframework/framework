//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Real = System.Double;
using Integer = System.Int32;

namespace Arcane.MatVec
{
  public struct VectorIndexer
  {
    internal RealArrayView m_values;
    //private Vector m_vector;
    
    public Real this[int i]
    {
      get { return m_values[i]; }
      set
      { 
#if ARCANE_CHECK
        if (Double.IsNaN(value) || Double.IsInfinity(value))
          throw new ApplicationException("Can not set NAN or Infinite value");
#endif
		m_values[i] = value;
      }
    }
    public RealArrayView Values { get { return m_values; } set { m_values = value; } }
    public Integer Size { get { return m_values.Size; } }
    public VectorIndexer(RealArrayView values)
    {
      m_values = values;
    }
  }
  
  /// <summary>
  /// Vecteur
  /// </summary>
  public class Vector : IDisposable
  {
    private RealArray m_values;
    public RealArrayView Values { get { return m_values.View; } }
    public RealConstArrayView ConstValues { get { return m_values.ConstView; } }

    public Integer Size { get { return m_values.Length; } }

    /// <summary>
    /// Construit un vecteur de \a size elements.
    /// Chaque element est initialise a zero.
    /// </summary>
    public Vector(Integer size)
    {
      m_values = new RealArray(size);
      FillZero();
    }
    
    ~Vector()
    {
      Console.WriteLine("Missing call to Arcane.MatVec.Vector.Dispose()");
      _Dispose(false);
    }
    
    public void Dispose()
    {
      _Dispose(true);
      GC.SuppressFinalize(this);
    }
    
    private void _Dispose(bool disposing)
    {
      if (disposing)
        m_values.Dispose();
    }
    
    /// <summary>
    /// Copie les valeurs du vecteur rhs dans l'instance.
    /// Les deux vecteurs doivent avoir la meme taille
    /// </summary>
    /// <param name="rhs">
    /// Le vecteur a copier <see cref="Vector"/>
    /// </param>
    public void Copy(Vector rhs)
    {
      m_values.Copy(rhs.m_values.ConstView);
    }
    
    /// <summary>
    /// Imprime le vecteur sur le flux o
    /// </summary>
    /// <param name="o">
    /// A <see cref="TextWriter"/>
    /// </param>
    public void Dump(TextWriter o)
    {
      RealConstArrayView v = ConstValues;
      for( Int32 row=0, n=m_values.Length; row<n; ++row ){
        o.Write("({0})={1} ",row,v[row]);
      }
    }
    
    /// <summary>
    /// Remplit le vecteur avec la valeur \a v
    /// </summary>
    public void Fill(Real v)
    {
      m_values.Fill(v);
    }

    /// <summary>
    /// Remplit le vecteur avec la valeur 0.0
    /// </summary>
    public void FillZero()
    {
      //TODO: verifier si possible d'utiliser memset
      m_values.Fill(0.0);
    }
    
    private VectorIndexer m_indexer;
    public VectorIndexer Indexer
    {
      get
      {
        m_indexer.Values = m_values.View;
        return m_indexer;
      }
    }
    
    public VectorIndexer GetIndexer(Integer begin,Integer size)
    {
      return new VectorIndexer(m_values.View.SubView(begin,size));
    }
  }
}
