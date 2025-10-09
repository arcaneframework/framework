// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
  Gestion des interfaces de Arcane.

  La notion d'interface n'existant pas en C++, toutes les classes C++ sont
  converties par SWIG en classe C#. Cependant, les classes virtuelles pures
  de Arcane qui commencent par 'I' comme 'IDataReader' ou 'IDataWriter'
  peuvent être considérées comme des interfaces au sens du C#.
  
  SWIG a le support des interfaces pour le C# via l'include 'swiginterface.i'.
  Dans ce cas, pour une interface C++ 'IToto', SWIG va générer une interface
  'IToto' et la classe C# 'IToto_INTERNAL' pour le wrapping de la classe C++.

  On définit la macro ARCANE_DECLARE_INTERFACE(namespace_name,interface_name)
  pour pouvoir wrapper une classe C++ comme une interface. Par exemple,
  pour wrapper la classe 'IDataReader', il suffit d'utiliser la macro
  comme suit:

  ARCANE_DECLARE_INTERFACE(Arcane,IDataReader)

  NOTE: Comme cette macro conduit au renommage de classe, il faut l'utiliser
  avant d'inclure le fichier d'en-tête contenant la définition de l'interface
  wrappée. L'idéal est donc de déclarer les interfaces en début de module.
*/ 
#if defined(SWIGCSHARP)
%include swiginterface.i

// La version 4.0 de SWIG a eu une erreur dans le typemap pour les interfaces: il manque le '.Handle'.
%define ARCANE_OVERRIDE_INTERFACE_TYPEMAP(CTYPE)
%typemap(csin) CTYPE, CTYPE & "$csinput.GetInterfaceCPtr().Handle"
%typemap(csdirectorout) CTYPE, CTYPE *, CTYPE *const&, CTYPE [], CTYPE & "$cscall.GetInterfaceCPtr().Handle"
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define ARCANE_DECLARE_INTERFACE(NAMESPACE_NAME,INTERFACE_NAME)
%rename(INTERFACE_NAME##_INTERNAL) NAMESPACE_NAME::INTERFACE_NAME;
%feature("interface", name=#INTERFACE_NAME) NAMESPACE_NAME::INTERFACE_NAME;
INTERFACE_TYPEMAPS(NAMESPACE_NAME::INTERFACE_NAME)

ARCANE_OVERRIDE_INTERFACE_TYPEMAP(NAMESPACE_NAME::INTERFACE_NAME)
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Ancienne macro 
%define ARCANE_DECLARE_INTERFACE_OLD(NAMESPACE_NAME,INTERFACE_NAME)
%rename(INTERFACE_NAME##_INTERNAL) NAMESPACE_NAME::INTERFACE_NAME;
%enddef

#endif // SWIGCSHARP
