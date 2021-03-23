//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif
using Real = System.Double;

namespace Arcane
{
  public class CaseOptionServiceContainer<InterfaceType>
    : ICaseOptionServiceContainer where InterfaceType : class
  {
    InterfaceType[] m_instances = new InterfaceType[0];

    public InterfaceType[] Values
    {
      get { return m_instances; }
    }

    public override void Allocate(int size)
    {
      m_instances = new InterfaceType[size];
    }

    public override int NbElem()
    {
      return m_instances.Length;
    }

    public override bool TryCreateService(int index, IServiceFactory2 factory, ServiceBuildInfoBase opt)
    {
      InterfaceType it = _tryCreateOneService(factory,opt);
      if (it!=null){
        m_instances[index] = it;
        return true;
      }
      Console.WriteLine("WARNING: TRY CREATE SERVICE NOT FOUND\n");
      return false;
    }

    public override bool HasInterfaceImplemented(IServiceFactory2 factory)
    {
      // TODO Implémenter cette méthode.
      // (Comme elle ne sert que pour afficher les infos sur les fabriques disponibles ce
      // n'est pas indispensable)
      return false;
    }

    InterfaceType _tryCreateOneService(IServiceFactory2 factory,ServiceBuildInfoBase opt)
    {
      Console.WriteLine("CHECK IS VALID SERVICE 2");
      HandleRef r = IServiceFactory2.getCPtr(factory);
      // Pour l'instant, on ne supporte que les services ecrits en C#
      // Via swig, \a factory est un objet cree par le wrapper.
      // Pour connaitre a quel fabrique cela correspond, il faut parcourir
      // celles enregistrees en C# et comparer les pointeurs C++
      // correspondants qui eux doivent identiques.
      foreach(IServiceFactory2 sf in ArcaneMain.DotNetFactories){
        HandleRef r2 = IServiceFactory2.getCPtr(sf);
        bool is_same = ((IntPtr)r==(IntPtr)r2);
        Console.WriteLine("CHECK: SF={0} ARG={1} ?={2} sf={3}",r2,r,is_same,sf);
        if (!is_same)
          continue;

        if (sf==null)
          continue;
        ServiceInstanceRef si_ref = sf.CreateServiceInstance(opt);
        if (si_ref==null)
          continue;
        Console.WriteLine("ServiceInstanceRef = '{0}'",si_ref);
        IServiceInstance si = si_ref.Get();
        Console.WriteLine("ServiceInstance = '{0}'",si);
        if (si==null)
          continue;
        ExternalRef handle = si._internalDotNetHandle();
        Console.WriteLine("HANDLE = '{0}'",handle);
        if (handle==null)
          continue;
        object dotnet_service = handle.GetReference();
        CSharpServiceInstance csi = dotnet_service as CSharpServiceInstance;
        Console.WriteLine("CSharpServiceInstance = '{0}'",csi);
        if (csi==null)
          continue;
        object my_service = csi.Instance();
        Console.WriteLine("Service = '{0}'",my_service);

        InterfaceType instance = my_service as InterfaceType;
        if (instance==null)
          throw new InvalidCastException(String.Format("Can not convert '{0}' to interface type '{1}'",
                                                       my_service,typeof(InterfaceType)));
        return instance;
      }
      return null;
    }
  }
}
