// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneSwigUtils.h                                           (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires pour le wrapping via Swig.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_WRAPPER_CORE_ARCANESWIGUTILS_H
#define ARCANE_WRAPPER_CORE_ARCANESWIGUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// NOTE: Ce fichier ne doit pas être inclus par swig via un '%include'.
// Il doit uniquement être utilisé dans le fichier 'C++' généré.

#include "NumericWrapper.h"

namespace Arcane
{
inline String
fromCSharpCharPtr(const char* ptr)
{
  if (ptr)
    return String(Arccore::StringView(ptr));
  return String();
}
inline StringView
fromCSharpCharPtrToStringView(const char* ptr)
{
  if (ptr)
    return Arccore::StringView(ptr);
  return StringView();
}
}

#include "arcane/BasicService.h"
#include "arcane/ServiceFactory.h"

namespace Arcane
{
  // Wrapper pour les variables.
  template<typename ItemType,typename DataType> class MeshVariableScalarRefT_Wrapper
  {
    MeshVariableScalarRefT_Wrapper(){}
  };
  // Wrapper pour les variables tableaux aux mailles
  template<typename ItemType,typename DataType> class MeshVariableArrayRefT_Wrapper
  { 
    MeshVariableArrayRefT_Wrapper(){}
  };
  // Wrapper pour les services.
  // Il faut faire une classe spéciale dont le premier héritage est l'interface car
  // SWIG ne wrappe que cela.
  template<typename InterfaceType>
  class BasicServiceWrapping : public InterfaceType, public BasicService
  {
   public:
    BasicServiceWrapping(const ServiceBuildInfo& sbi) : BasicService(sbi)
    {
    }
   public:
    static Internal::IServiceFactory2*
    createTemplateFactory(IServiceInfo* si,Internal::IServiceInterfaceFactory<InterfaceType>* functor)
    {
      return new Internal::ServiceFactory2TV2<InterfaceType>(si,functor);
    }
  };
  // Wrapper pour les services (Version 3).
  // Il faut faire une classe spéciale dont le premier héritage est l'interface car
  // SWIG ne wrappe que cela.
  template<typename InterfaceType>
  class BasicServiceWrapping3 : public BasicService, public InterfaceType
  {
   public:
    BasicServiceWrapping3(const ServiceBuildInfo& sbi) : BasicService(sbi)
    {
    }
   public:
    static Internal::IServiceFactory2*
    createTemplateFactory(IServiceInfo* si,Internal::IServiceInterfaceFactory<InterfaceType>* functor)
    {
      return new Internal::ServiceFactory2TV2<InterfaceType>(si,functor);
    }
  };
}

#include "arcane/utils/Exception.h"
#include "arcane/utils/OStringStream.h"

namespace Arcane
{
  //! Fonction utilitaire pour récupérer une éventuelle exception pour la passer entre le C++ et le C#
  template<typename Lambda> inline
  void _doSwigTryCatch(const Lambda& lambda)
  {
    try{
      lambda();
    }
    catch (const Arcane::Exception& ex) {
      OStringStream ostr;
      ostr() << ex;
      String s = ostr.str();
      std::string s2 = s.localstr();
      SWIG_CSharpSetPendingException(SWIG_CSharpApplicationException, s2.c_str());
    }
    catch (const std::exception& e) {
      SWIG_CSharpSetPendingException(SWIG_CSharpApplicationException, e.what());
    }
    catch(...) {
      SWIG_CSharpSetPendingException(SWIG_CSharpApplicationException, "Unknown");
    }
  }
}

#endif
