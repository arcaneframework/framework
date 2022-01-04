//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif
using Real = System.Double;

namespace Arcane
{
  public struct TraceAccessor
  {
    // Les valeurs suivantes doivent être cohérentes avec celles
    // de Trace::eMessageType du C++ (fichier utils/Trace.h)
    const int MESSAGE_Normal = 0;
    const int MESSAGE_Info = 1;
    const int MESSAGE_Warning = 2;
    const int MESSAGE_Error = 3;
    const int MESSAGE_Log = 4;
    const int MESSAGE_Fatal = 5;
    const int MESSAGE_ParallelFatal = 6;
    const int MESSAGE_Debug = 7;
    const int MESSAGE_Null = 8;

    ITraceMng m_trace_mng;
    public TraceAccessor(ITraceMng tm)
    {
      m_trace_mng = tm;
    }

    public void Info(string str)
    {
      m_trace_mng.PutTrace(str,MESSAGE_Info);
    }

    public void Info(string format,params object[] objs)
    {
      m_trace_mng.PutTrace(String.Format(format,objs),MESSAGE_Info);
    }

    public void Fatal(string str)
    {
      m_trace_mng.PutTrace(str,MESSAGE_Fatal);
    }

    public void Fatal(string format,params object[] objs)
    {
      m_trace_mng.PutTrace(String.Format(format,objs),MESSAGE_Fatal);
    }

    public void Warning(string str)
    {
      m_trace_mng.PutTrace(str,MESSAGE_Warning);
    }

    public void Warning(string format,params object[] objs)
    {
      m_trace_mng.PutTrace(String.Format(format,objs),MESSAGE_Warning);
    }
  }
}
