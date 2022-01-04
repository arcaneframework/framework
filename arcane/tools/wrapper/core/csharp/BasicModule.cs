//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;

#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif
using Real = System.Double;

namespace Arcane
{
  internal class CSharpBasicModule : BasicModule_INTERNAL
  {
    public CSharpBasicModule(ModuleBuildInfo mbi) : base(mbi)
    {
    }
    public override bool IsGarbageCollected()
    {
      //Console.WriteLine("MODULE IS GARBAGE COLLECTED");
      return true;
    }
  }

  public class BasicModule
  {
    private BasicModule_INTERNAL m_cpp_module;
    private List<EntryPoint> m_entry_points = new List<EntryPoint>();

    readonly protected VariableScalarInt32 m_global_iteration;
    readonly protected VariableScalarReal m_global_deltat;
    readonly protected VariableScalarReal m_global_time;
    readonly protected VariableScalarReal m_global_old_time;
    readonly protected VariableScalarReal m_global_old_deltat;
    TraceAccessor m_trace_accessor;
    public TraceAccessor Trace { get { return m_trace_accessor; } }

    public Int32 GlobalIteration { get { return m_global_iteration.Value; } }
    public Real GlobalDeltaT { get { return m_global_deltat.Value; } } 
    public Real GlobalTime { get { return m_global_time.Value; } } 
    public Real GlobalOldDeltaT { get { return m_global_old_deltat.Value; } } 
    public Real GlobalOldTime { get { return m_global_old_time.Value; } } 

    public BasicModule(ModuleBuildInfo mbi)
    {
      m_cpp_module = new CSharpBasicModule(mbi);
      ISubDomain sd = mbi.SubDomain();
      m_trace_accessor = new TraceAccessor(sd.TraceMng());
      m_global_iteration = new VariableScalarInt32(new VariableBuildInfo(sd,"GlobalIteration",0));
      m_global_deltat = new VariableScalarReal(new VariableBuildInfo(sd,"GlobalDeltaT",0));
      m_global_time = new VariableScalarReal(new VariableBuildInfo(sd,"GlobalTime",0));
      m_global_old_deltat = new VariableScalarReal(new VariableBuildInfo(sd,"GlobalOldDeltaT",0));
      m_global_old_time = new VariableScalarReal(new VariableBuildInfo(sd,"GlobalOldTime",0));
    }
    
    public static implicit operator IModule(BasicModule m)
    {
      return m.m_cpp_module; 
    }

    public IModule Module { get { return m_cpp_module; } }

    public IMesh DefaultMesh() { return m_cpp_module.DefaultMesh(); }
    
    public ISubDomain SubDomain() { return m_cpp_module.SubDomain(); }

    public IParallelMng ParallelMng() { return m_cpp_module.ParallelMng(); }

    public CellGroup AllCells() { return DefaultMesh().AllCells(); }
    public NodeGroup AllNodes() { return DefaultMesh().AllNodes(); }
    public FaceGroup AllFaces() { return DefaultMesh().AllFaces(); }

    public CellGroup OwnCells() { return DefaultMesh().OwnCells(); }
    public NodeGroup OwnNodes() { return DefaultMesh().OwnNodes(); }
    public FaceGroup OwnFaces() { return DefaultMesh().OwnFaces(); }

    protected void _AddEntryPoint(string name,IFunctor.FunctorDelegate callback, string where, int property)
    {
      EntryPoint e = new EntryPoint(this,name,callback,where,property);
      m_entry_points.Add(e);
    }
  }
}
