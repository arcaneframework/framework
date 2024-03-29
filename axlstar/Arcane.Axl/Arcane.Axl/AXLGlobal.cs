//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AXLGlobal.cs                                                (C) 2000-2022 */
/*                                                                           */
/* Déclarations générales pour le package AXL.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using System;

namespace Arcane.Axl
{
  //! Type de service disponible
  public enum ServiceType
  {
    ST_Application,
    ST_Session,
    ST_SubDomain,
    ST_Mesh,
    ST_CaseOption,
    ST_Unknown
  }
  public enum Property
  {
    PNone,
    PAutoLoadEnd,
    PAutoLoadBegin,
  }
  public class InternalErrorException : Exception
  {
    public InternalErrorException(string msg) : base(msg) { }
  }

  public sealed class GlobalContext
  {
    private static readonly GlobalContext _instance = new GlobalContext();
 
    private GlobalContext() { }
 
    public static GlobalContext Instance
    {
        get
        {
           return _instance;
        }
    }
    
    private bool noRestore = false;
    private bool verbose = true;
    
    public bool NoRestore
    {
       get { return noRestore; }
       set { noRestore = value; }
    }
    
    public bool Verbose
    {
       get { return verbose; }
       set { verbose = value; }
    }
  }
}
