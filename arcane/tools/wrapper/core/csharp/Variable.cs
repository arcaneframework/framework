//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Runtime.InteropServices;

#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif
using Real = System.Double;

namespace Arcane
{
  public class Variable
  {
    protected IVariable m_internal_var;
   
    protected Variable()
    {
    }

    //! Nom de la variable
    public string Name { get { return m_internal_var.Name(); } }
  }
  /*
   * Classe gérant le callback de notification lorsqu'une 
   * variable sur le maillage change de taille.
   *
   * Pour gérer correctement le callback en présence de
   * garbage collecting, il est nécessaire d'avoir une
   * instance distincte de MeshVariableRef pour gérer
   * le fonctor passé au C++. Cette classe gère cela
   * et garde une référence faible sur la variable
   * C# associée ce qui permet à cette dernière d'être
   * collectée par le GC tout en gardant le callback valide.
   */
  internal class VariableUpdateNotifier
  {
    WeakReference m_mesh_variable;
    MeshVariableRef.UpdateDelegate saved_del;
    IntPtr m_notify_ptr;

    static object global_lock = new object();

    internal VariableUpdateNotifier(MeshVariableRef var)
    {
      saved_del = this._OnSizeChanged;
      m_mesh_variable = new WeakReference(var,true);
    }
    
    void _OnSizeChanged()
    {
      // Ne fait l'appel que si la référence est toujours vivante.
      if (m_mesh_variable.IsAlive){
        MeshVariableRef mr = (MeshVariableRef)m_mesh_variable.Target;
        mr._DirectOnSizeChanged();
      }
    }

    internal void Register(HandleRef var)
    {
			lock(global_lock){
        m_notify_ptr = MeshVariableRef._AddChangedDelegate(var,saved_del);
      }
    }

    internal void Unregister()
    {
      // Comme cette methode est appelee dans le Dispose de MeshVariableRef
      // et que ce dispose peut etre appele depuis n'importe quel thread,
      // il faut proteger les appels par un verrou.
      // En théorie, il faut un verrou par variable C++ mais comme
      // ce n'est pas très pratique à obtenir, on fait un verrou global.
      lock(global_lock){
        MeshVariableRef._RemoveChangedDelegate(m_notify_ptr,saved_del);
      }
    }
  }

}
