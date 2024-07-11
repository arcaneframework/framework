//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections;
using System.Diagnostics;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Runtime.CompilerServices;

// Pour accéder à ItemSharedInfo. A supprimer lorsqu'il n'y aura plus
// besoin de ItemInternal
[assembly: InternalsVisibleToAttribute("Arcane.Cea.Materials")]

namespace Arcane
{
  public static class ArcaneMain
  {
    static void _WriteException(Exception e, string message, int level = 0)
    {
      Console.WriteLine ("{3}: Exception type={0}, message={1}, stack_trace={2}",
                         e.GetType (), e.Message, e.StackTrace, message);
      Exception ie = e.InnerException;
      if (ie != null)
        _WriteException (ie, message, level + 1);
    }

    static StringList m_args = new StringList();
    static List<ServiceFactoryInfo> m_created_builders = new List<ServiceFactoryInfo>();
    static List<IServiceFactory2> m_created_factories2 = new List<IServiceFactory2>();

    static List<ModuleFactoryReference> m_created_module_factories = new List<ModuleFactoryReference>();

    static internal IList<IServiceFactory2> DotNetFactories { get { return m_created_factories2; } }
    static int m_verbose_level;
    static public int VerboseLevel { get { return m_verbose_level; } }
    static public bool IsVerbose { get { return m_verbose_level>=1; } }
    static public bool IsVerboseLevel2 { get { return m_verbose_level>=2; } }
    static public bool HasDotNetWrapper { get { return ArcaneMain_INTERNAL.HasDotNetWrapper(); } }
    // Met le callback dans un champ statique pour ne pas qu'il soit collecté par le GC.
    static ArcaneMain_INTERNAL._GarbageCollectorDelegate m_garbage_collector_callback;

    static ArcaneMain()
    {
      string s = Environment.GetEnvironmentVariable("ARCANE_DEBUG_DOTNET");
      if (s=="1" || s=="true" || s=="TRUE")
        m_verbose_level = 1;
      if (s=="2")
        m_verbose_level = 2;
    }

    static void _PrintStructuresSize()
    {
      Debug.Write("Sizeof wrapped structures: ItemInternalConnectivityList: {0}",
                  Marshal.SizeOf(typeof(ItemInternalConnectivityList)));
      Debug.Write("Sizeof wrapped structures: ItemInternal: {0}",
                  Marshal.SizeOf(typeof(ItemInternal)));
      Debug.Write("Sizeof wrapped structures: ItemSharedInfo: {0}",
                  Marshal.SizeOf(typeof(ItemSharedInfo)));
    }

    static public void Initialize()
    {
      Debug.Write("NOTE: Using Verbose mode for '.Net' runtime");
      Debug.Write("NOTE: Using .NET version={0}", Environment.Version);
      _PrintStructuresSize();
      ArcaneMain_INTERNAL.SetHasGarbageCollector();
      m_garbage_collector_callback = CallGarbageCollector;
      ArcaneMain_INTERNAL._ArcaneWrapperCoreSetCallGarbageCollectorDelegate(m_garbage_collector_callback);
      ArcaneMain_INTERNAL.SetHasDotNETRuntime();
      ArcaneMain_INTERNAL.ArcaneInitialize();
      _LoadInternalModulesAndServices();
    }

    static public void Initialize(string [] args)
    {
      m_args = new StringList();
      foreach (string s in args) {
        m_args.Add(s);
      }
      Initialize();
    }

    static public void CallGarbageCollector()
    {
      Debug.Write("Calling garbage collector");
      for( int i=0; i<3; ++i ){
        GC.Collect();
        GC.WaitForPendingFinalizers();
      }
    }

    static internal void RegisterModule(string name, VersionInfo version, Type module_type)
    {
      Debug.Write("Registering module name={0} type={1}", name, module_type);
      ModuleBuilder mb = new ModuleBuilder(module_type, name);
      var mb_ref = IModuleFactory2Ref.CreateWithHandle(mb,ExternalRef.Create(mb));
      ModuleFactoryReference mf = new ModuleFactoryReference(mb_ref, false);
      m_created_module_factories.Add(mf);
      ArcaneMain_INTERNAL.AddModuleFactoryInfo(mf.Factory());
    }

    static internal void RegisterServiceWithFactory(string name, Type service_type, ServiceInfo si, GenericServiceFactory gsf)
    {
      Debug.Write("Registering service factory name={0} type={1} usage={2}", name, service_type, si.UsageType());
      ServiceBuilderWithFactory sfb = new ServiceBuilderWithFactory(si, gsf);
      m_created_builders.Add(sfb);
      ArcaneMain_INTERNAL.AddServiceFactoryInfo(sfb);
    }

    public delegate void ExecutionOverrideDelegate(IApplication app);
    class DotNetExecutionOverrideFunctor : IFunctor
    {
      ExecutionOverrideDelegate m_dotnet_functor;
      internal ArcaneMainExecutionOverrideFunctor m_arcane_functor;
      public DotNetExecutionOverrideFunctor(ExecutionOverrideDelegate functor)
      {
        m_dotnet_functor = functor;
      }

      public override void ExecuteFunctor()
      {
        m_dotnet_functor(m_arcane_functor.Application());
      }
    }
    static DotNetExecutionOverrideFunctor m_override_functor;

    static public void SetExecuteOverrideFunctor(ExecutionOverrideDelegate functor)
    {
      var d = new DotNetExecutionOverrideFunctor(functor);
      var x = new ArcaneMainExecutionOverrideFunctor(d);
      d.m_arcane_functor = x;
      m_override_functor = d;
      ArcaneMain_INTERNAL.SetExecuteOverrideFunctor(x);
    }

    static public int Exec(string code_name, VersionInfo code_version)
    {
      ApplicationInfo app_info = new ApplicationInfo(m_args, code_name, code_version);
      return Exec(app_info);
    }

    static public ApplicationInfo CreateApplicationInfo(string code_name, VersionInfo code_version)
    {
      ApplicationInfo app_info = new ApplicationInfo(m_args, code_name, code_version);
      return app_info;
    }

    static public int Exec(ApplicationInfo app_info)
    {
      int r = ArcaneMain_INTERNAL.ArcaneMain(app_info);
      Cleanup();
      return r;
    }

    class DirectMethodInfo
    {
#pragma warning disable 0649
      public MethodInfo m_method;
#pragma warning restore 0649
    }

    static Type _GetTypeFromLoadededAssemblies(string class_name)
    {
      foreach (Assembly a in AppDomain.CurrentDomain.GetAssemblies()) {
        Type x = a.GetType(class_name);
        if (x != null)
          return x;
      }
      return null;
    }

    // Récupère un pointeur sur la méthode de nom \a method_name de la classe \a class_name.
    static MethodInfo _GetExecMethod(string class_name, string method_name)
    {
      Type x = _GetTypeFromLoadededAssemblies(class_name);
      if (x == null)
        throw new ApplicationException(String.Format("Can not find class named '{0}'", class_name));
      BindingFlags b = BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod | BindingFlags.FlattenHierarchy;
      MethodInfo method = x.GetMethod(method_name, b);
      if (method == null) {
        var s = String.Format("Can not find static method named '{0}' in class '{1}'", method_name, class_name);
        throw new ApplicationException(s);
      }
      return method;
    }
    //! Attend pour s'attacher à un débugger
    static public void WaitForDebugger(bool print_process_id)
    {
      if (print_process_id){
        Process p = Process.GetCurrentProcess();
        Console.WriteLine($"Waiting for debug to attach PID={p.Id}...");
      }
      // Boucle tant qu'on n'est pas attaché et lorsque c'est le cas
      // génère un breakpoint.
      // Pour ne pas surcharger le CPU, ne teste que une fois par seconde.
      for(int i=0; i<10000; ++i ){
        if (Debugger.IsAttached){
          Debugger.Break();
          break;
        }
        Thread.Sleep(1000);
      }
    }

    static public int Run()
    {
      string s = Environment.GetEnvironmentVariable("ARCANE_DOTNET_WAITFORDEBUGGER");
      if (s=="1")
        WaitForDebugger(true);

      var dotnet_info = DotNetRuntimeInitialisationInfo;

      // Charge l'assembly eventuellement spécifiée
      string assembly_name = dotnet_info.MainAssemblyName();
      var helper = new AssemblyLoaderHelper();
      helper.LoadSpecifiedAssembly(assembly_name);

      // Exécute la méthode spécifiée si demandée.
      MethodInfo method_to_exec = null;
      string class_name = dotnet_info.ExecuteClassName();
      string method_name = dotnet_info.ExecuteMethodName();
      if (!String.IsNullOrEmpty(class_name)) {
        if (String.IsNullOrEmpty(method_name))
          method_name = "Main";
        method_to_exec = _GetExecMethod(class_name, method_name);
        Debug.Write("Trying to execute method '{0}' in class '{1}'", method_name, class_name);
      }
      if (method_to_exec != null) {
        // La méthode récupérée peut avoir deux prototypes:
        //   1. int function(ArcaneSimpleExecutor executor).
        //   2. void function().
        // Dans le premier cas, il s'agit d'une exécution directe et après appel à la méthode
        // on arrête le code. Dans le second, il s'agit juste d'une méthode d'initialisation et
        // ensuite on exécute le code normalement par la méthode ArcaneMain::run().
        ParameterInfo[] method_params = method_to_exec.GetParameters();
        Type return_type = method_to_exec.ReturnType;
        Debug.Write("Execute method return type = '{0}' nb_param={1}", return_type, method_params.Length);
        if (return_type == typeof(int) && method_params.Length == 1
         && method_params [0].ParameterType == typeof(ArcaneSimpleExecutor)) {
          int ret_value = ArcaneSimpleExecutor.Run((ArcaneSimpleExecutor executor) => {
            // TODO: encapsuler pour récupérer les exceptions
            return (int)method_to_exec.Invoke(null, new object [] { executor });
          });
          return ret_value;
        }
        else if (return_type == typeof(void) && method_params.Length == 0) {
          method_to_exec.Invoke(null, null);
        }
        else {
          string s1 = "Invalid signature for execute method '{0}'."+
            " Valid signatures are 'int f(ArcaneSimpleExecutor)' or 'void f()'.";
          throw new ApplicationException(String.Format(s1,method_name));
        }
      }

      // Attention de bien faire l'initialisation après la lecture
      // de l'assembly sinon les services et modules de cette dernière ne
      // seront pas chargés.
      Initialize();
      return _Run2();
    }

    static int _Run2()
    {
      int r = 0;
      if (HasDotNetWrapper)
        r = ArcaneMain_INTERNAL.ArcaneMain(DefaultApplicationInfo);
      else
        r = ArcaneMain_INTERNAL.Run();
      Cleanup();
      return r;
    }

    static public ApplicationInfo DefaultApplicationInfo
    {
      get {
        return ArcaneMain_INTERNAL.DefaultApplicationInfo();
      }
    }

    static public DotNetRuntimeInitialisationInfo DotNetRuntimeInitialisationInfo
    {
      get {
        return ArcaneMain_INTERNAL.DefaultDotNetRuntimeInitialisationInfo();
      }
    }

    static public IAssemblyLoader AssemblyLoader
    {
      get { return AssemblyLoaderHelper.Loader; }
      set { AssemblyLoaderHelper.Loader = value; }
    }

    static public void Cleanup()
    {
      m_created_module_factories.Clear();
      CallGarbageCollector();
    }

    //! Indique si \a aname est un nom valide pour une assembly contenant des services ou des modules
    static bool _IsValidArcaneAssembly(string aname)
    {
      // Pas besoin de parcourir ce genre d'assembly.
      if (aname=="mscorlib")
        return false;
      if (aname.StartsWith("System."))
        return false;
      if (aname.StartsWith("Mono."))
        return false;
      return true;
    }

    /*!
     * \brief Parcours les assembly et charge les modules ou services qui y sont definis.
     *
     * La recherche se fait grâce aux attributs 'Module' ou 'Service'.
     */
    static void _LoadInternalModulesAndServices()
    {
      Debug.Write("Loading internal modules and services");
      HashSet<string> m_already_loaded = new HashSet<string>();
      foreach(Assembly a in AppDomain.CurrentDomain.GetAssemblies()){
        string full_name = a.GetName().FullName;
        if (m_already_loaded.Contains(full_name)){
          Debug.Write($"Skip assembly '{full_name}' because it has already been analysed");
          continue;
        }
        m_already_loaded.Add(full_name);
        if (_IsValidArcaneAssembly(a.GetName().Name)){
          _LoadModuleAndServiceFromAssembly(a);
        }
        else
          Debug.Write($"Skip assembly '{full_name}' because it is filtered by name");
      }
      Debug.Write("End Loading internal modules and services");
    }

    static readonly Type typeof_module = typeof(Arcane.ModuleAttribute);
    static readonly Type typeof_service = typeof(Arcane.ServiceAttribute);
    static readonly BindingFlags service_build_info_flags = BindingFlags.Public|BindingFlags.Static|
                                                            BindingFlags.InvokeMethod|BindingFlags.FlattenHierarchy;

    //! Regarde si le type \a t est un service ou un module et si c'est le cas l'enregistre
    static void _CheckModuleOrServiceType(Type t)
    {
      Debug.Write(2,"Checking Type {0}",t);

      // Regarde s'il s'agit d'un module (attribut 'ModuleAttribute')
      object[] objs = t.GetCustomAttributes(typeof_module,false);
      if (objs!=null && objs.Length>0){
        Debug.Write("FOUND MODULE TYPE WITH ATTRIBUTE: {0}",t);
        ModuleAttribute mattr = (ModuleAttribute)objs[0];
        VersionInfo vi = new VersionInfo(mattr.Version);
        RegisterModule(mattr.Name,vi,t);
        return;
      }

      // Regarde s'il s'agit d'un service (attribut 'ServiceAttribute')
      objs = t.GetCustomAttributes(typeof_service,false);
      if (objs==null || objs.Length<=0)
        return;
      Debug.Write("FOUND SERVICE TYPE WITH ATTRIBUTE: {0}",t);
      ServiceAttribute sattr = (ServiceAttribute)objs[0];

      // Recherche la méthode de construction 'serviceInfoCreateFunction()' (générée à partir d'un fichier axl).
      MethodInfo build_info_method = t.GetMethod("serviceInfoCreateFunction",service_build_info_flags);
      // Recherche la méthode de construction de la fabrique.
      MethodInfo create_factory_method = t.GetMethod("CreateFactory",service_build_info_flags);
      Debug.Write("BuilInfo method = {0}",build_info_method);
      Debug.Write("CreateFactory method = {0}",create_factory_method);
      ServiceInfo si = null;
      ServiceType service_usage_type = sattr.Type;
      string service_name = sattr.Name;

      if (build_info_method!=null){
        Debug.Write("HAS BUILD_INFO_METHOD");
        si = (ServiceInfo)build_info_method.Invoke(null,new object[]{ service_name });
      }
      else{
        si = Arcane.ServiceInfo.Create(service_name,(int)service_usage_type);
        si.AddImplementedInterface(sattr.InterfaceType.FullName);
        si.SetAxlVersion(1);
        si.SetDefaultTagName(service_name);
      }

      if (si==null){
        Console.WriteLine("WARNING: can not load service type='{0}' because no 'serviceInfoCreateFunction' is provided",t);
        return;
      }

      if (create_factory_method==null){
        Console.WriteLine("WARNING: can not load service type='{0}' because no 'CreateFactory' method is provided",t);
        return;
      }

      GenericServiceFactory gsf = new GenericServiceFactory(si,t);
      IServiceFactory2 sf2 = (IServiceFactory2)create_factory_method.Invoke(null,new object[]{ gsf });
      if (sf2==null){
        Console.WriteLine("WARNING: Factory created is null");
        return;
      }
      si.AddFactory(sf2);
      m_created_factories2.Add(sf2);
      var sfactory = new GenericSingletonServiceFactory(sf2);
      gsf.SingletonFactory = sfactory;
      si.SetSingletonFactory(sfactory);
      RegisterServiceWithFactory(service_name,t,si,gsf);
    }

    static void _LoadModuleAndServiceFromAssembly(Assembly a)
    {
      // On ne peut pas appeler 'a.Location' sur une assembly dynamique
      if (a.IsDynamic)
        Debug.Write("Reading dynamic assembly name={0}",a);
      else
        Debug.Write("Reading assembly name={0} path={1}",a,a.Location);

      Type[] atypes = null;
      try{
        atypes = a.GetTypes();
      }
      catch(Exception ex){
        _WriteException(ex,$"Exception during getting types of assembly '{a}'");
        return;
      }

      foreach(Type t in atypes){
        try{
          _CheckModuleOrServiceType(t);
        }
        catch(Exception ex){
          _WriteException(ex,$"Exception during loading of type '{t}'");
        }
      }
    }
  }
}
