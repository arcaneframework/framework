// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
// Wrapper Arcane vers le C#
%module(directors="1", allprotected="1") ArcaneDotNet

// Ces deux macros permettent que les constructeurs des
// classes générées par SWIG et qui prennent le pointeur C++
// en arguments soient publiques. Sans cela elles sont 'internal'
// ce qui ne permet pas de les utiliser si on définit d'autres
// wrapping C# (par exemple le wrapping HDF5)
SWIG_CSBODY_PROXY([System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)] public,
                  [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)] public,
                  SWIGTYPE)
SWIG_CSBODY_TYPEWRAPPER([System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)] public,
                        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)] public,
                        [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)] public, SWIGTYPE)

%{
#include "core/ArcaneSwigCoreInclude.h"
#include "ArcaneSwigUtils.h"

using namespace Arcane;
using namespace Arcane::Accelerator;
%}

#define ARCANE_RESTRICT
#define ARCANE_BEGIN_NAMESPACE  namespace Arcane {
#define ARCANE_END_NAMESPACE    }

#if SWIG_VERSION >= 0x040000
#define SWIG4
#endif

#define SWIG_DISPOSING csdisposing
#define SWIG_DISPOSE csdispose
#define SWIG_DISPOSE_DERIVED csdispose_derived

namespace Arccore
{
  class IFunctor;
  namespace Internal
  {
    class ExternalRef;
  }
}
namespace Arcane
{
  using Arccore::IFunctor;
  namespace Internal
  {
    using Arccore::Internal::ExternalRef;
  }
}

#define ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(a)

#define ARCANE_DEPRECATED_112
#define ARCANE_DEPRECATED_114
#define ARCANE_DEPRECATED_116
#define ARCANE_DEPRECATED_118
#define ARCANE_DEPRECATED_120
#define ARCANE_DEPRECATED_122
#define ARCANE_DEPRECATED_200
#define ARCANE_DEPRECATED_220
#define ARCANE_DEPRECATED_240
#define ARCANE_DEPRECATED_260
#define ARCANE_DEPRECATED_2018
#define ARCANE_DEPRECATED_2018_R(a)
#define ARCCORE_DEPRECATED_2019(a)
#define ARCCORE_DEPRECATED_2020(a)
#define ARCCORE_DEPRECATED_2021(a)
#define ARCANE_DEPRECATED_REASON(a)
#define ARCANE_DEPRECATED_REASON(a)
#define ARCANE_DEPRECATED_LONG_TERM(a)
#define ARCANE_NOEXCEPT

#define ARCCORE_BASE_EXPORT
#define ARCCORE_COMMON_EXPORT
#define ARCANE_EXPR_EXPORT
#define ARCANE_CORE_EXPORT
#define ARCANE_UTILS_EXPORT
#define ARCANE_STD_EXPORT
#define ARCANE_DEPRECATED
#define ARCANE_DATATYPE_EXPORT
#define ARCANE_IMPL_EXPORT
#define ARCANE_ACCELERATOR_CORE_EXPORT

#define ARCCORE_HOST_DEVICE

%ignore operator();
%ignore operator[];

// Permet de convertir les 'void*' en 'IntPtr'
%apply void* VOID_INT_PTR { void* };

%include stdint.i
#if defined(SWIGWORDSIZE64)
%typemap(imtype) long,               const long &               "long"
%typemap(imtype) unsigned long,      const unsigned long &      "ulong"
%typemap(cstype) long,               const long &               "long"
%typemap(cstype) unsigned long,      const unsigned long &      "ulong"
%typemap(csout, excode=SWIGEXCODE) long, const long &
{
  long ret = $imcall;$excode
  return ret;
}
%typemap(csout, excode=SWIGEXCODE) unsigned long, const unsigned long &
{
  ulong ret = $imcall;$excode
  return ret;
}
%typemap(csvarout, excode=SWIGEXCODE2) long, const long &
%{
 get {
    long ret = $imcall;$excode
    return ret;
 }
%}
%typemap(csvarout, excode=SWIGEXCODE2) unsigned long, const unsigned long &
%{
 get {
   ulong ret = $imcall;$excode
   return ret;
 }
 %}
#endif

namespace Arcane
{
typedef double Real;
#if defined(ARCANE_64BIT)
typedef Int64 Integer;
#else
typedef Int32 Integer;
#endif
typedef int64_t Int64;
typedef int32_t Int32;
typedef int16_t Int16;
typedef uint32_t UInt32;
typedef uint64_t UInt64;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Macro pour générer un type POD
// - CPP_TYPE est le type C++
// - CSHARP_TYPE est le type C#.
// Pour que cela fonctionne, il faut que la macro soit utilisée avant que le
// type ne soit utilisé.
// Les types 'MutableMemoryView' et 'ConstMemoryView' sont des exemples d'utilisation.
%define ARCANE_SWIG_GENERATE_POD_TYPE(CPP_TYPE,CSHARP_TYPE)
%typemap(csinterfaces) CPP_TYPE "";
%typemap(csbody) CPP_TYPE %{ %}
%typemap(SWIG_DISPOSING, methodname="Dispose", methodmodifiers="private") CPP_TYPE ""
%typemap(SWIG_DISPOSE, methodname="Dispose", methodmodifiers="private") CPP_TYPE ""
%typemap(csclassmodifiers) CPP_TYPE "public struct"
%typemap(csattributes) CPP_TYPE "[StructLayout(LayoutKind.Sequential)]"
%typemap(cstype) CPP_TYPE "CSHARP_TYPE"
%typemap(ctype, out="CPP_TYPE",
	 directorout="CPP_TYPE",
	 directorin="CPP_TYPE") CPP_TYPE "CPP_TYPE"
%typemap(imtype) CPP_TYPE "CSHARP_TYPE"
%typemap(csin) CPP_TYPE "$csinput"
%typemap(csout) CPP_TYPE { return $imcall;  }
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Gestion des interfaces de Arcane.
// Ce fichier doit être inclus avant toute classe utilisant des
// interfaces de Arcane.
%include Interface.i

%include InterfacesDeclaration.i

 /*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Indique les classes pouvant être dérivées en C#
%feature("director") AbstractObserver;

// Positionne la liste des imports au début de chaque fichier
%typemap(csimports) SWIGTYPE
%{
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
#if ARCANE_64BIT
using Integer = System.Int64;
using IntegerConstArrayView = Arcane.Int64ConstArrayView;
using IntegerArrayView = Arcane.Int64ArrayView;
#else
using Integer = System.Int32;
using IntegerConstArrayView = Arcane.Int32ConstArrayView;
using IntegerArrayView = Arcane.Int32ArrayView;
#endif
using Real = System.Double;
using Arcane;
%}

namespace Arccore
{
  class String;
  class StringView;
};

namespace Arcane
{
  using Arccore::String;
  using Arccore::StringView;
  class IServiceFactory;
  class ServiceRegisterer;
  class Real3;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Renomme les méthodes pour qu'elles commencent par une majuscule
// suivant la convention du C#.
%rename("%(firstuppercase)s", %$isfunction, %$ismember) "";

// Ne renomme pas les méthodes de 'std::vector' car sinon le template
// 'std_vector.i' ne fonctionne pas car il utilise les fonctions wrappées
%rename("capacity", regextarget=1, fullname=1) "std::vector<.*>::capacity$";
%rename("getitem", regextarget=1, fullname=1) "std::vector<.*>::getitem$";
%rename("setitem", regextarget=1, fullname=1) "std::vector<.*>::setitem$";
%rename("item", regextarget=1, fullname=1) "std::vector<.*>::item$";
%rename("size", regextarget=1, fullname=1) "std::vector<.*>::size$";
%rename("reserve", regextarget=1, fullname=1) "std::vector<.*>::reserve$";
%rename("getitemcopy", regextarget=1, fullname=1) "std::vector<.*>::getitemcopy$";
%rename("empty", regextarget=1, fullname=1) "std::vector<.*>::empty$";

// Fait de même avec 'std::map'
%rename("getitem", regextarget=1, fullname=1) "std::map<.*>::getitem$";
%rename("setitem", regextarget=1, fullname=1) "std::map<.*>::setitem$";
%rename("size", regextarget=1, fullname=1) "std::map<.*>::size$";
%rename("empty", regextarget=1, fullname=1) "std::map<.*>::empty$";
%rename("create_iterator_begin", regextarget=1, fullname=1) "std::map<.*>::create_iterator_begin$";
%rename("get_next_key", regextarget=1, fullname=1) "std::map<.*>::get_next_key$";
%rename("destroy_iterator", regextarget=1, fullname=1) "std::map<.*>::destroy_iterator$";

/*---------------------------------------------------------------------------*/

%rename("_enumerator", regextarget=1, fullname=1) "Arcane::.*::enumerator$";

%rename("IsNull", regextarget=1, fullname=1) "Arcane::.*::null$";
%rename("_internal", regextarget=1, fullname=1) "Arcane::.*::internal$";

%rename("_current", regextarget=1, fullname=1) "Arcane::EnumeratorT<.*>::current$";
%rename("_count", regextarget=1, fullname=1) "Arcane::Collection<.*>::count$";
%rename("_empty", regextarget=1, fullname=1) "Arcane::Collection<.*>::empty$";
%rename("_current", fullname=1) "Arcane::VariableCollectionEnumerator::current";
%rename("_count", fullname=1) "Arcane::VariableCollection::count";
%rename("_empty", fullname=1) "Arcane::VariableCollection::empty";
%rename("_values", regextarget=1, fullname=1) "Arcane::CaseOptionMulti.*::values";
%rename("$ignore", regextarget=1, fullname=1) "Arcane::VariableCollection::(clear|add)";
%rename("_count", fullname=1) "Arcane::VariableCollection::count";

// Les types 'Array2View' et 'ConstArray2View' ne sont pas wrappés
%rename("$ignore", regextarget=1, fullname=1) "Arcane::IArray2DataT<.*>::view";

%rename("_value", regextarget=1, fullname=1) "Arcane::VariableRefScalarT<.*>::value$";
%rename("_asArray", regextarget=1, fullname=1) "Arcane::ItemVariable.*::asArray";
%rename("_finalize", fullname=1) "Arcane::IArcaneMain::finalize";
%rename("_finalize", fullname=1) "Arcane::ArcaneMain::finalize";

// Renomme cette méthode car il y a une classe de même nom
%rename("_acceleratorRuntimeInitialisationInfo", fullname=1) "Arcane::ArcaneMain::acceleratorRuntimeInitialisationInfo";

// Supprime le wrapping des méthodes qui sont obsolètes dans Arcane
%rename("$ignore", fullname=1, regextarget=1) "Arcane::ServiceBuilder<.*>::createInstance";
%rename("$ignore", fullname=1) "Arcane::ApplicationInfo::m_argc";
%rename("$ignore", fullname=1) "Arcane::ApplicationInfo::m_argv";
%rename("$ignore", fullname=1) "Arcane::ApplicationInfo::commandLineArgc";
%rename("$ignore", fullname=1) "Arcane::ApplicationInfo::commandLineArgv";
%rename("$ignore", fullname=1) "Arcane::IApplication::configRootElement";
%rename("$ignore", fullname=1) "Arcane::IApplication::userConfigRootElement";
%rename("$ignore", fullname=1) "Arcane::ITimeHistoryMng::removeCurveWriter";
%rename("$ignore", fullname=1) "Arcane::ITimeHistoryMng::addCurveWriter";
%rename("$ignore", fullname=1) "Arcane::ISubDomain::doInitModules";
%rename("$ignore", fullname=1) "Arcane::ISubDomain::mesh";
%rename("$ignore", fullname=1) "Arcane::ISubDomain::findMesh";
%rename("$ignore", fullname=1) "Arcane::ISubDomain::meshDimension";
%rename("$ignore", fullname=1) "Arcane::IItemFamily::addItems";
%rename("$ignore", fullname=1) "Arcane::IItemFamily::removeItems";
%rename("$ignore", fullname=1) "Arcane::IItemFamily::exchangeItems";
%rename("$ignore", fullname=1, regextarget=1) "Arcane::IItemFamily::max.*PerItem";
%rename("$ignore", fullname=1) "Arcane::IVariable::write";
%rename("$ignore", fullname=1) "Arcane::IVariable::read";
%rename("$ignore", fullname=1) "Arcane::ICaseOptions::subDomain";
%rename("$ignore", fullname=1) "Arcane::ICaseOptions::mesh";
%rename("$ignore", fullname=1) "Arcane::ICaseOptions::read";
%rename("$ignore", fullname=1) "Arcane::CaseOptions::subDomain";
%rename("$ignore", fullname=1) "Arcane::CaseOptions::mesh";
%rename("$ignore", fullname=1) "Arcane::CaseOptions::read";
%rename("$ignore", fullname=1) "Arcane::CaseOptionServiceImpl::read";
%rename("$ignore", fullname=1) "Arcane::IApplication::serviceFactoryInfos";
%rename("$ignore", fullname=1) "Arcane::IMainFactory::createTimeStats";
%rename("$ignore", fullname=1) "Arcane::IItemFamily::variableMaxSize";
%rename("$ignore", fullname=1) "Arcane::IItemFamily::mergeItems";
%rename("$ignore", fullname=1) "Arcane::IItemFamily::getMergedItemLID";
%rename("$ignore", fullname=1) "Arcane::ISerializedData::dimensions";
%rename("$ignore", fullname=1) "Arcane::CaseOptionBase::subDomain";
%rename("$ignore", regextarget=1, fullname=1) "Arcane::IArrayDataT<.*>::value";
%rename("$ignore", regextarget=1, fullname=1) "Arcane::IArrayDataT<.*>::cloneTrue";
%rename("$ignore", regextarget=1, fullname=1) "Arcane::IArrayDataT<.*>::cloneTrueEmpty";
%rename("$ignore", regextarget=1, fullname=1) "Arcane::IArray2DataT<.*>::value";
%rename("$ignore", regextarget=1, fullname=1) "Arcane::IArray2DataT<.*>::cloneTrue";
%rename("$ignore", regextarget=1, fullname=1) "Arcane::IArray2DataT<.*>::cloneTrueEmpty";

// Les méthodes suivantes sont obsolètes mais utilisées par des exemples
// donc il ne faut pas encore les supprimer
#if NOT_YET
%rename("$ignore", fullname=1) "Arcane::ISerializedData::buffer";
%rename("$ignore", fullname=1) "Arcane::ISerializedData::setBuffer";
%rename("$ignore", fullname=1) "Arcane::IMeshModifier::addCells";
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Ignore certaines méthodes qui génèrent des types qui ne sont pas wrappés.

%ignore Arcane::IMesh::findItemFamilyModifier;
%ignore Arcane::IVariableMng::onVariableAdded;
%ignore Arcane::IVariableMng::onVariableRemoved;
%ignore Arcane::IItemFamily::itemsNewOwner;
%ignore Arcane::IItemFamily::removeItems2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Avec Swig 4.0.0, il manque le nom du namespace Avec le $classname ce qui
// peut rendre le C# généré ambigu s'il existe dans la classe une méthode
// ayant le même nom qu'un type C#. Pour éviter cela on préfixe le nom
// de la classe du namespace Arcane
%define ARCANE_SWIG_OVERRIDE_GETCPTR(NAME,NAMESPACE_NAME)
%typemap(csin) NAME "NAMESPACE_NAME . $&csclassname.getCPtr($csinput)"
%typemap(csin) NAME*, NAME&, NAME&&, NAME[] "NAMESPACE_NAME . $csclassname.getCPtr($csinput)"
%typemap(csdirectorout) NAME "NAMESPACE_NAME . $&csclassname.getCPtr($cscall).Handle"
%typemap(csdirectorout) NAME*, NAME&, NAME&& "NAMESPACE_NAME . $csclassname.getCPtr($cscall).Handle"
%enddef

ARCANE_SWIG_OVERRIDE_GETCPTR(Arcane::ItemGroup,Arcane)
ARCANE_SWIG_OVERRIDE_GETCPTR(Arcane::MeshHandle,Arcane)
ARCANE_SWIG_OVERRIDE_GETCPTR(Arcane::VersionInfo,Arcane)
ARCANE_SWIG_OVERRIDE_GETCPTR(Arcane::ApplicationInfo,Arcane)
ARCANE_SWIG_OVERRIDE_GETCPTR(Arcane::ApplicationBuildInfo,Arcane)
ARCANE_SWIG_OVERRIDE_GETCPTR(Arcane::CommandLineArguments,Arcane)
ARCANE_SWIG_OVERRIDE_GETCPTR(Arcane::MeshPartInfo,Arcane)
ARCANE_SWIG_OVERRIDE_GETCPTR(Arcane::ItemGroupT<Arcane::Node>,Arcane)
ARCANE_SWIG_OVERRIDE_GETCPTR(Arcane::ItemGroupT<Arcane::Edge>,Arcane)
ARCANE_SWIG_OVERRIDE_GETCPTR(Arcane::ItemGroupT<Arcane::Face>,Arcane)
ARCANE_SWIG_OVERRIDE_GETCPTR(Arcane::ItemGroupT<Arcane::Cell>,Arcane)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Fonctions pour enregistrer un delegate qui permet d'appeler depuis le C++
// les routines C# qui gèrent le GarbageCollector.
%typemap(cscode) Arcane::ArcaneMain
%{
  internal delegate void _GarbageCollectorDelegate();
  [DllImport("$dllimport")]
  static internal extern IntPtr _ArcaneWrapperCoreSetCallGarbageCollectorDelegate(_GarbageCollectorDelegate d);
%}
%{
  extern "C" ARCANE_UTILS_EXPORT void
  _ArcaneSetCallGarbageCollectorDelegate(void (*func)());
  extern "C" ARCANE_EXPORT void
  _ArcaneWrapperCoreSetCallGarbageCollectorDelegate(void (*func)())
  {
    _ArcaneSetCallGarbageCollectorDelegate(func);
  }
%}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%rename EntryPoint EntryPoint_INTERNAL;
%rename ArcaneMain ArcaneMain_INTERNAL;
%rename ITraceMng ITraceMng_INTERNAL;
%rename ArcaneSimpleExecutor ArcaneSimpleExecutor_INTERNAL;

%include StringView.i
%include String.i

%include Numeric.i

%include ArrayView.i
%include MemoryView.i

%feature("director") Arcane::IBinaryMathFunctor;
%include arcane/utils/IMathFunctor.h
%template(IRealRealToRealMathFunctor) Arcane::IBinaryMathFunctor<double,double,double>;
%template(IRealRealToReal3MathFunctor) Arcane::IBinaryMathFunctor<double,double,Arcane::Real3>;
%template(IRealReal3ToRealMathFunctor) Arcane::IBinaryMathFunctor<double,Arcane::Real3,double>;
%template(IRealReal3ToReal3MathFunctor) Arcane::IBinaryMathFunctor<double,Arcane::Real3,Arcane::Real3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Wrapping de ItemInternalArrayView (et ItemInternalList car c'est la même classe)

%template(ItemInternalArrayView) Arcane::ArrayView<ItemInternal*>;
%typemap(ctype, out="Arcane::ItemInternalArrayView",
         directorout="Arcane::ItemInternalArrayView",
         directorin="Arcane::ItemInternalArrayView") ItemInternalArrayView "Arcane.ItemInternalArrayView"
%typemap(cstype) ItemInternalArrayView "Arcane.ItemInternalArrayView"
%typemap(imtype) ItemInternalArrayView "Arcane.ItemInternalArrayView"
%typemap(csin) ItemInternalArrayView "$csinput"
%typemap(out) ItemInternalArrayView
%{
   $result = $1;
%}
%typemap(csout) ItemInternalArrayView {
    ItemInternalArrayView ret = $imcall;$excode
    return ret;
  }
%typemap(csdirectorin) ItemInternalArrayView "$iminput"
%typemap(csdirectorout) ItemInternalArrayView "$cscall"

namespace Arcane
{
  using ItemInternaList = Arcane::ArrayView<ItemInternal*>;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Wrapping de ItemInfoListView. La structure C# correspondante est dans csharp/ItemInternal.cs

%typemap(ctype, out="Arcane::ItemInfoListView",
         directorout="Arcane::ItemInfoListView",
         directorin="Arcane::ItemInfoListView") Arcane::ItemInfoListView "Arcane.ItemInfoListView"
%typemap(cstype) Arcane::ItemInfoListView "Arcane.ItemInfoListView"
%typemap(imtype) Arcane::ItemInfoListView "Arcane.ItemInfoListView"
%typemap(csin) Arcane::ItemInfoListView "$csinput"
%typemap(out) Arcane::ItemInfoListView
%{
   $result = $1;
%}
%typemap(csout) Arcane::ItemInfoListView {
    ItemInfoListView ret = $imcall;$excode
    return ret;
  }
%typemap(csdirectorin) Arcane::ItemInfoListView "$iminput"
%typemap(csdirectorout) Arcane::ItemInfoListView "$cscall"

 /*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%typemap(csinterfaces) Arcane::ITraceMng "Arcane.ITraceMng";
namespace Arcane
{
  class ITraceMng
  {
   protected:
    virtual ~ITraceMng(){}
   public:
    virtual void putTrace(const String& str,int message_type) =0;
  };        
}

// Ajoute méthode de construction de CommandListArguments à partir d'un 'string[]' du C#.
%typemap(cscode) Arcane::CommandLineArguments %{
  static public CommandLineArguments Create(string[] args)
  {
    StringList slist = new StringList();
    foreach(string s in args){
      slist.Add(s);
    }
    return new CommandLineArguments(slist);
  }
%}

%include Ref.i
%include arccore/common/CommandLineArguments.h
%include arcane/utils/VersionInfo.h
%include arcane/utils/ApplicationInfo.h

%include DotNetObject.i

namespace Arcane
{
class IEntryPoint
{
 public:
  static const char* const WComputeLoop;
  static const char* const WBuild;
  static const char* const WInit;
  static const char* const WContinueInit;
  static const char* const WStartInit;
  static const char* const WRestore;
  static const char* const WOnMeshChanged;
  static const char* const WOnMeshRefinement;
  static const char* const WExit;
  enum
  {
    PNone = 0,
    PAutoLoadBegin = 1,
    PAutoLoadEnd   = 2
  };
 public:
  virtual ~IEntryPoint() {}
  virtual String name() const =0;
};
}

%include arcane/core/datatype/DataTypes.h
%include arcane/core/IBase.h
%include arcane/core/ISubDomain.h
%include arcane/core/SharedVariable.h
%include arcane/core/ArcaneTypes.h
%include arcane/core/IApplication.h
%include arcane/core/IDirectory.h
%include arcane/core/ItemGroupImpl.h
%include arcane/core/SharedReference.h
%include arccore/base/IObserver.h
%include arccore/base/Observer.h
%include arccore/base/IObservable.h
%include arcane/utils/ArrayShape.h
%include arcane/core/IMainFactory.h
%include arcane/core/ApplicationBuildInfo.h
%include arcane/core/DotNetRuntimeInitialisationInfo.h
%include arccore/common/accelerator/AcceleratorRuntimeInitialisationInfo.h

namespace Arcane
{
class EntryPointBuildInfo
{
 public:
  EntryPointBuildInfo(IModule* module,const String& name,
                      IFunctor* caller,const String& where,int property,
                      bool is_destroy_caller);
};

class EntryPoint : public IEntryPoint
{
 public:
  virtual ~EntryPoint();
 private:
  // Marque le constructeur comme privé pour interdire de créer
  // des instances.
  EntryPoint();
 public:
  static EntryPoint* create(const EntryPointBuildInfo& bi);
  virtual String name() const;
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define ARCANE_STD_EXHANDLER
%exception {
  ::Arcane::_doSwigTryCatch([&]{ $action });
}
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_STD_EXHANDLER
%ignore Arcane::IItemFamily::synchronizeVariable;
%include arcane/core/IItemFamily.h
%include arcane/impl/ArcaneMain.h
%include arcane/impl/ArcaneSimpleExecutor.h
%exception;

%include core/DataVisitor.i
%include core/Collection.i
%include core/Variables.i
%include core/Service.i
%include core/ItemGroup.i
%include core/ItemPairGroup.i

ARCANE_STD_EXHANDLER
%include arcane/core/IApplication.h
%include arcane/core/ISession.h
%exception;

ARCANE_STD_EXHANDLER
%include core/Mesh.i
%exception;

%include arcane/core/IVariableMng.h
%include arcane/core/IVariable.h
%include arcane/core/IArcaneMain.h
%include arcane/core/ITimeLoopMng.h
%include arcane/core/datatype/ScalarVariant.h
%include arcane/core/StdNum.h

%ignore Arcane::XmlNode::end;
%ignore Arcane::XmlNode::begin;
%include arcane/core/XmlNode.h
%include arcane/core/XmlNodeList.h

%include core/CaseOption.i
%include core/Parallel.i

%include arcane/core/ITimeHistoryMng.h
%include arcane/core/CommonVariables.h

%typemap(csbase) Arcane::AssertionException "System.Exception"

%include arcane/core/ArcaneException.h

ARCANE_STD_EXHANDLER
%include arcane/core/MeshReaderMng.h
%exception;

%include core/InterfacesInclude.i
