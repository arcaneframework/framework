//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections;
using System.Diagnostics;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Arcane
{
  // Ce type doit correspondre avec le type C++ correspondant
  [Flags]
  public enum ServiceType
  {
    ST_None        = 0,
    ST_Application = 1,
    ST_Session     = 2,
    ST_SubDomain   = 4,
    ST_CaseOption  = 8
  }

  /*!
   * \brief Gère une instance d'un service C# qui n'a pas d'interface C++ correspondante.
   *
   * Un tel service peut donc uniquement être appelé depuis un code C#.
   */
  public class CSharpServiceInstance : DotNetServiceInstance
  {
    IServiceInfo m_service_info;
    object m_instance;
    public CSharpServiceInstance(object o,IServiceInfo si)
      : base (si)
    {
      m_service_info = si;
      m_instance = o;
    }
    public override IServiceInfo ServiceInfo() { return m_service_info; }
    public object Instance() { return m_instance; }
  }

  /*!
   * \brief Fabrique pour un service C# utilisant un fichier axl mais
   * qui n'a pas d'interface C++ correspondante.
   */
  public class AxlGeneratedServiceFactory : AbstractServiceFactory
  {
    GenericServiceFactory m_generic_factory;

    public AxlGeneratedServiceFactory(GenericServiceFactory gs)
    {
      m_generic_factory = gs;
    }

    //! Retourne le IServiceInfo associé à cette fabrique.
    public override IServiceInfo ServiceInfo()
    {
      return m_generic_factory.ServiceInfo;
    }
    
    public override ServiceInstanceRef CreateServiceInstance(ServiceBuildInfoBase sbi)
    {
      object o = m_generic_factory.CreateInstanceDirect(sbi);
      if (o==null)
        return null;
      var x = new CSharpServiceInstance(o,m_generic_factory.ServiceInfo);
      var h = ExternalRef.Create(x);
      x.SetDotNetHandle(h);
      ServiceInstanceRef x2 = ServiceInstanceRef.CreateWithHandle(x,ExternalRef.Create(x));
      return x2;
    }
  }

  public class GenericSingletonServiceFactory : SingletonServiceFactoryBase
  {
    IServiceFactory2 m_factory;
    public GenericSingletonServiceFactory(IServiceFactory2 factory)
      : base(factory.ServiceInfo())
    {
      m_factory = factory;
    }
    protected override ServiceInstanceRef _createInstance(ServiceBuildInfoBase sbi, IServiceInstanceAdder instance_adder)
    {
      ServiceInstanceRef sr = m_factory.CreateServiceInstance(sbi);
      instance_adder.AddInstance(sr);
      return sr;
    }
  }

  public class GenericServiceFactory
  {
    // Ce champs sert juste pour conserver le pointeur vers la fabrique de l'interface afin
    // qu'elle ne soit pas collectée par le GC
    public object InterfaceFactoryObject { set; get; }
    internal GenericSingletonServiceFactory SingletonFactory { set; get; }

    Arcane.ServiceType m_service_type;
    public Arcane.ServiceType ServiceType { get { return m_service_type; } }

    ConstructorInfo m_constructor_info;
    IServiceInfo m_service_info;

    public GenericServiceFactory(IServiceInfo si,Type type)
    {
      m_service_info = si;
      if (type==null)
        throw new ArgumentException("null type");

      m_service_type = (ServiceType)si.UsageType();

      m_service_info = si;
      Type[] arg_types = new Type[]{ typeof(ServiceBuildInfo) };
      ConstructorInfo c = type.GetConstructor(BindingFlags.Instance|BindingFlags.Public,null,
                                              CallingConventions.HasThis,arg_types,null);
      Debug.Write("GEN_SF C INFO c_info={0}",c);
      m_constructor_info = c;
      if (c==null){
        string ex_str = String.Format("class '{0}' has no valid constructor. it is not a service",type);
        throw new ArgumentException(ex_str);
      }
    }

    public object CreateInstanceDirect(ServiceBuildInfoBase sbib)
    {
      ServiceType sbi_type = (ServiceType)sbib.CreationType();
      if ((ServiceType & sbi_type)!=sbi_type)
        return null;
      Debug.Write("GenericServiceFactory: CreateInstanceDirect name={0}",m_service_info.LocalName());
      if (ArcaneMain.IsVerboseLevel2)
        Console.WriteLine("Trace: {0}",new System.Diagnostics.StackTrace());
      return _CreateInstance(new ServiceBuildInfo(m_service_info,sbib));
    }

    public IServiceInfo ServiceInfo { get { return m_service_info; } }

    private object _CreateInstance(ServiceBuildInfo sbi)
    {
      object[] args = new object[] { sbi };
      object new_service = m_constructor_info.Invoke(args);
      Debug.Write("TRY CAST o={0} type={1}",new_service,new_service.GetType());

      ISubDomain sd = sbi.SubDomain();
      object s = new_service;
      Debug.Write("CREATE SERVICE 2 s='{0}'",s);
      Debug.Write("CREATE SERVICE 3 's={0}' 'hash_code={1}'",s,s.GetHashCode());
      Debug.Write("RETURN");
      return s;
    }
  }

  internal class ModuleBuilder : IModuleFactory2
  {
    ConstructorInfo m_constructor_info;
    string m_name;
    ServiceInfo m_service_info;

    public ModuleBuilder(Type type, string name)
    {
      if (type == null)
        throw new ArgumentException("null type");
      if (String.IsNullOrEmpty(name))
        throw new ArgumentException("null name");
      m_name = name;
      Type [] arg_types = new Type [] { typeof(ModuleBuildInfo) };
      ConstructorInfo c = type.GetConstructor(BindingFlags.Instance | BindingFlags.Public, null,
                                              CallingConventions.HasThis, arg_types, null);
      Debug.Write("C INFO c_info={0}", c);
      m_constructor_info = c;
      if (c == null)
        throw new ArgumentException(String.Format("class '{0}' has no valid constructor. it is not a module",
                                                  type));
      ServiceInfo si = Arcane.ServiceInfo.Create(name, 0);
      si.SetDefaultTagName(name);
      m_service_info = si;
    }

    public override IModuleRef CreateModuleInstance(ISubDomain sd, MeshHandle mesh_handle)
    {
      Debug.Write("CREATE MODULE name={0}", m_name);
      ModuleBuildInfo mbi = new ModuleBuildInfo(sd, mesh_handle, m_name);
      object [] args = new object [] { mbi };
      BasicModule m = (BasicModule)m_constructor_info.Invoke(args);
      return IModuleRef.CreateWithHandle(m,ExternalRef.Create(m));
    }

    public override void InitializeModuleFactory(ISubDomain sd)
    {
    }

    public override string ModuleName()
    {
      return m_name;
    }
    public override IServiceInfo ServiceInfo()
    {
      return m_service_info;
    }
  }

  // Pour l'instant, on ne peut créer que des services du jeu de données.
  internal class ServiceBuilderWithFactory : ServiceFactoryInfo
  {
    GenericServiceFactory m_generic_factory;
    public ServiceBuilderWithFactory(IServiceInfo si, GenericServiceFactory generic_factory)
    : base(si)
    {
      m_generic_factory = generic_factory;
    }

    IService _ToIService(object o)
    {
      Debug.Write("TRY TO CREATE ISERVICE!!!");
      if (o == null)
        return null;
      Debug.Write("TRY TO CONVERT TO ISERVICE!!!");
      return null;
    }
  }
}
