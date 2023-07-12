// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
%feature("director") ServiceFactoryInfo;
%feature("director") BasicModule;
%feature("director") BasicService;
%feature("director") AbstractService;
%feature("director") IFunctor;
%feature("director") IFunctorWithArgumentT;
%feature("director") IServiceInstance;
%feature("director") Arcane::Internal::IServiceFactory2;
%feature("director") AbstractServiceFactory;
%feature("director") Arcane::Internal::IServiceInterfaceFactory;
%feature("director") Arcane::Internal::IServiceFactory2T;
%feature("director") Arcane::Internal::SingletonServiceFactoryBase;
%feature("director") IModuleFactoryInfo;
%feature("director") IModuleFactory2;

%rename BasicModule BasicModule_INTERNAL;

%typemap(cscode) Arccore::IFunctor %{
  public delegate void FunctorDelegate();
  private FunctorDelegate m_functor;
  public FunctorDelegate Functor { get { return m_functor; } }
  public class Wrapper : IFunctor
  {
    public Wrapper(FunctorDelegate functor)
    {
      m_functor = functor;
    }
    public override void ExecuteFunctor()
    {
      if (m_functor!=null)
        m_functor();
    }
  }
%}

%typemap(cscode) Arcane::ModuleFactory
%{
  public delegate IModule ModuleFactoryFunc(ISubDomain sd,IMesh mesh);

  static public ModuleFactory Create(IServiceInfo si,ModuleFactoryFunc func)
  {
    return null;
  }
%}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TRES IMPORTANT: Il ne faut pas detruire les fabriques de service
// creee en C# car ca fait planter a la fin de l'execution.
// Cela ne provoque de toutes facon pas de fuite memoire car cela
// n'est utilisé qu'avant le main
// Il faudra essayer de faire plus propre par la suite
%typemap(SWIG_DISPOSE_DERIVED, methodname="Dispose", methodmodifiers="public") Arcane::Internal::ServiceFactoryBase "{}"
%typemap(SWIG_DISPOSING) Arcane::Internal::ServiceFactoryBase %{ %}

// TRES IMPORTANT: Il ne faut pas detruire les fabriques de service
// creee en C# car ca fait planter a la fin de l'execution.
// Cela ne provoque de toutes facon pas de fuite memoire car cela
// n'est utilisé qu'avant le main
// Il faudra essayer de faire plus propre par la suite
%typemap(SWIG_DISPOSE_DERIVED, methodname="Dispose", methodmodifiers="public") Arcane::Internal::ServiceFactoryInfo "public virtual void Dispose(){}"
%typemap(SWIG_DISPOSING) Arcane::Internal::ServiceFactoryInfo %{ %}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Derived proxy classes
// A partir de Swig 2.0, il faut utiliser *_SWIGUpcast au lieu de *Upcast
%typemap(csbody_derived) Arcane::BasicService
%{
  private HandleRef swigCPtr;
  readonly protected VariableScalarInt32 m_global_iteration;
  readonly protected VariableScalarReal m_global_deltat;
  readonly protected VariableScalarReal m_global_time;
  readonly protected VariableScalarReal m_global_old_time;
  readonly protected VariableScalarReal m_global_old_deltat;
  TraceAccessor m_trace_accessor;
  public TraceAccessor Trace { get { return m_trace_accessor; } }

  internal $csclassname(IntPtr cPtr, bool cMemoryOwn) : base($imclassname.$csclassname_SWIGUpcast(cPtr), cMemoryOwn)
  {
    swigCPtr = new HandleRef(this, cPtr);
    ISubDomain sd = SubDomain();
    m_trace_accessor = new TraceAccessor(sd.TraceMng());
    m_global_iteration = new VariableScalarInt32(new VariableBuildInfo(sd,"GlobalIteration",0));
    m_global_deltat = new VariableScalarReal(new VariableBuildInfo(sd,"GlobalDeltaT",0));
    m_global_time = new VariableScalarReal(new VariableBuildInfo(sd,"GlobalTime",0));
    m_global_old_deltat = new VariableScalarReal(new VariableBuildInfo(sd,"GlobalOldDeltaT",0));
    m_global_old_time = new VariableScalarReal(new VariableBuildInfo(sd,"GlobalOldTime",0));
  }

  internal static HandleRef getCPtr($csclassname obj) {
    return (obj == null) ? new HandleRef(null, IntPtr.Zero) : obj.swigCPtr;
  }
%}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define SWIG_ARCANE_REGISTER_MESHACCESSOR_FUNCTIONS
  public:
   Integer nbCell() const;
   Integer nbFace() const;
   Integer nbNode() const;
   VariableNodeReal3& nodesCoordinates() const;
   NodeGroup allNodes() const;
   FaceGroup allFaces() const;
   CellGroup allCells() const;
   FaceGroup outerFaces() const;
   NodeGroup ownNodes() const;
   CellGroup ownCells() const;
   FaceGroup ownFaces() const;
   IMesh* mesh() const;
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%ignore Arcane::ServiceBuilder::getServicesNames;
%ignore Arcane::IServiceInfo::factories;

%include arcane/core/IService.h
%include arcane/core/IServiceInfo.h
%include arcane/core/ServiceProperty.h
%include arcane/core/IServiceFactory.h
%include arcane/core/ServiceFactory.h
%include arcane/core/ServiceBuilder.h
%include arcane/core/ServiceInstance.h
%include arcane/core/ModuleBuildInfo.h
%include arcane/core/ServiceBuildInfo.h
%include arcane/core/IModule.h
%include arcane/core/AbstractModule.h
%include arcane/core/BasicModule.h
%include arcane/core/IEntryPoint.h

namespace Arccore
{
class IFunctor
{
 public:
  virtual ~IFunctor(){}
  virtual void executeFunctor() =0;
};
template<typename ArgType>
class IFunctorWithArgumentT
{
 public:
  virtual ~IFunctorWithArgumentT() {}
  virtual void executeFunctor(ArgType arg) =0;
};
}

%include arcane/core/IModuleFactory.h
%include arcane/core/ModuleFactory.h

%include arcane/core/ServiceInfo.h

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%feature("director") Arcane::BasicServiceWrapping;
%feature("director") Arcane::BasicServiceWrapping3;

namespace Arcane
{
  class AbstractService : public IService
  {
    protected: AbstractService(const ServiceBuildInfo&);
    public: virtual void build() {}
  };
}

namespace Arcane
{
  class BasicService : public AbstractService
  {
    protected: BasicService(const ServiceBuildInfo&);
    public: ISubDomain* subDomain();
    SWIG_ARCANE_REGISTER_MESHACCESSOR_FUNCTIONS
  };
}
namespace Arcane
{
  class BasicUnitTest : public BasicService
  {
    public: BasicUnitTest(const ServiceBuildInfo& sbi);
    public: virtual void initializeTest();
    public: virtual void executeTest();
    public: virtual void finalizeTest();
  };
  template<typename InterfaceType>
  class BasicServiceWrapping : public InterfaceType
  {
    public:
     BasicServiceWrapping(const ServiceBuildInfo& sbi);
     static Internal::IServiceFactory2*
     createTemplateFactory(IServiceInfo* si,Internal::IServiceInterfaceFactory<InterfaceType>* ptr);
     SWIG_ARCANE_REGISTER_MESHACCESSOR_FUNCTIONS
  };
  template<typename InterfaceType>
  class BasicServiceWrapping3 : public InterfaceType
  {
    public:
     BasicServiceWrapping(const ServiceBuildInfo& sbi);
     static Internal::IServiceFactory2*
     createTemplateFactory(IServiceInfo* si,Internal::IServiceInterfaceFactory<InterfaceType>* ptr);
     SWIG_ARCANE_REGISTER_MESHACCESSOR_FUNCTIONS
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// -------------------- MACRO (Version 2) pour générer le wrapping C# d'un service.
// Obsolète: utiliser la version 3

%define SWIG_ARCANE_DEFINE_SERVICE_INTERFACE2(NAMESPACE_NAME,INTERFACE_NAME)
%typemap(cscode) Arcane::BasicServiceWrapping<NAMESPACE_NAME::INTERFACE_NAME>
%{
  public class MyServiceInterfaceFactory : Arcane.INTERFACE_NAME##_IServiceInterfaceFactory
  {
    GenericServiceFactory gen;
    public MyServiceInterfaceFactory(GenericServiceFactory gsf)
    {
      gen = gsf;
    }
    public override INTERFACE_NAME##_Ref CreateReference(Arcane.ServiceBuildInfo sbi)
    {
      var x = (INTERFACE_NAME##_INTERNAL)gen.CreateInstanceDirect(sbi);
      return INTERFACE_NAME##_Ref . CreateWithHandle(x, ExternalRef.Create(x));
    }
  }

  static public IServiceFactory2 CreateFactory(GenericServiceFactory gsf)
  {
    Arcane.INTERFACE_NAME##_IServiceInterfaceFactory iif = new MyServiceInterfaceFactory(gsf);
    // Il faut conserver l'objet créé pour ne pas qu'il soit collecté par le GC
    gsf.InterfaceFactoryObject = iif;
    IServiceFactory2 f2 = CreateTemplateFactory(gsf.ServiceInfo,iif);
    return f2;
  }

%}
%template(INTERFACE_NAME##_WrapperService) Arcane::BasicServiceWrapping<NAMESPACE_NAME::INTERFACE_NAME>;
%template(INTERFACE_NAME##_IServiceInterfaceFactory) Arcane::Internal::IServiceInterfaceFactory<NAMESPACE_NAME::INTERFACE_NAME>;
%template(INTERFACE_NAME##_ServiceBuilder) Arcane::ServiceBuilder<NAMESPACE_NAME::INTERFACE_NAME>;
%template(INTERFACE_NAME##_Ref) Arcane::Ref<NAMESPACE_NAME::INTERFACE_NAME>;
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
  MACRO (Version 3) pour générer le wrapping C# d'un service.

  Cette macro ARCANE_SWIG_DEFINE_SERVICE permet de wrapper les informations
  nécessaires pour créer et utiliser un service en C# implémentant une
  interface donnée.

  La macro utilise trois paramètres:

  ARCANE_SWIG_DEFINE_SERVICE(namespace_name,interface_name,PROXY_CODE)

  - namespace_name: nom du namespace de l'interface. Il ne doit pas être nul.
  - interface_name: nom de l'interface.
  - PROXY_CODE: code C# indiquant la liste des méthodes de l'interface sous
  forme de méthode abstraite du C#. A terme, cela pourra peut-être être généré
  directement par Swig.

  Par exemple, pour wrapper l'interface 'Arcane::IDataReader':

  ARCANE_SWIG_DEFINE_SERVICE(Arcane,IDataReader,
                             public abstract void beginRead(VariableCollection vars);
                             public abstract void endRead();
                             public abstract string metaData();
                             public abstract void read(IVariable var, IData data);
                        );
*/
// Il faut dire à SWIG que notre classe 'BasicServiceWrapping3' dérive pas seulement de l'interface
// mais aussi de la classe Proxy (TODO: vérifier si cela est nécessaire)
%define ARCANE_SWIG_DEFINE_SERVICE(NAMESPACE_NAME,INTERFACE_NAME,PROXY_CODE)
%typemap(csattributes) Arcane::BasicServiceWrapping3<NAMESPACE_NAME::INTERFACE_NAME> %{ abstract %}
%typemap(cscode) Arcane::BasicServiceWrapping3<NAMESPACE_NAME::INTERFACE_NAME>
%{
  public class MyServiceInterfaceFactory : INTERFACE_NAME##_IServiceInterfaceFactory
  {
    GenericServiceFactory gen;
    public MyServiceInterfaceFactory(GenericServiceFactory gsf)
    {
      gen = gsf;
    }
    public override INTERFACE_NAME##_Ref CreateReference(Arcane.ServiceBuildInfo sbi)
    {
      var x = (INTERFACE_NAME)gen.CreateInstanceDirect(sbi);
      return INTERFACE_NAME##_Ref . CreateWithHandle(x, ExternalRef.Create(x));
    }
  }

  static public Arcane.IServiceFactory2 CreateFactory(Arcane.GenericServiceFactory gsf)
  {
    INTERFACE_NAME##_IServiceInterfaceFactory iif = new MyServiceInterfaceFactory(gsf);
    // Il faut conserver l'objet créé pour ne pas qu'il soit collecté par le GC
    gsf.InterfaceFactoryObject = iif;
    IServiceFactory2 f2 = CreateTemplateFactory(gsf.ServiceInfo,iif);
    return f2;
  }

  PROXY_CODE
%}
%template(INTERFACE_NAME##_WrapperService) Arcane::BasicServiceWrapping3<NAMESPACE_NAME::INTERFACE_NAME>;
%template(INTERFACE_NAME##_IServiceInterfaceFactory) Arcane::Internal::IServiceInterfaceFactory<NAMESPACE_NAME::INTERFACE_NAME>;
%template(INTERFACE_NAME##_ServiceBuilder) Arcane::ServiceBuilder<NAMESPACE_NAME::INTERFACE_NAME>;
%template(INTERFACE_NAME##_Ref) Arcane::Ref<NAMESPACE_NAME::INTERFACE_NAME>;
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%template(SingletonServiceInstanceRef) Arcane::Ref<Arcane::ISingletonServiceInstance>;
%template(IModuleRef) Arcane::Ref<Arcane::IModule>;
%template(IModuleFactory2Ref) Arcane::Ref<Arcane::IModuleFactory2>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
