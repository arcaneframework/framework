// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DomDeclaration.h                                            (C) 2000-2025 */
/*                                                                           */
/* Déclarations du DOM.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_DOMDECLARATION_H
#define ARCANE_CORE_DOMDECLARATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_BEGIN_NAMESPACE_DOM namespace dom {
#define ARCANE_END_NAMESPACE_DOM   }

#define ARCANE_BEGIN_NAMESPACE_DOMUTILS namespace domutils {
#define ARCANE_END_NAMESPACE_DOMUTILS   }

/*
 * Ces deux macros ne sont plus utilisées par Arcane, mais on les laisse
 * pour compatibilité avec les applications qui pourraient les utiliser.
 */
#define ARCANE_HAVE_DOM2
#define ARCANE_HAVE_DOM3

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::dom
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NodePrv;
class AttrPrv;
class ElementPrv;
class NamedNodeMapPrv;
class DocumentPrv;
class DocumentTypePrv;
class ImplementationPrv;
class CharacterDataPrv;
class TextPrv;
class NodeListPrv;
class DocumentFragmentPrv;
class CommentPrv;
class CDATASectionPrv;
class ProcessingInstructionPrv;
class EntityReferencePrv;
class EntityPrv;
class NotationPrv;
class DOMErrorPrv;
class DOMLocatorPrv;
class DOMWriterPrv;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Document;
class Node;
class DocumentFragment;
class NodeList;
class NamedNodeMap;
class CharacterData;
class Attr;
class Element;
class Text;
class Comment;
class CDATASection;
class DocumentType;
class Notation;
class Entity;
class EntityReference;
class ProcessingInstruction;

typedef unsigned short UShort;
typedef unsigned long ULong;

typedef unsigned long DOMTimeStamp;

class DOMWriter;
class DOMImplementationSource;
typedef void* DOMObject;
class UserDataHandler;
class DOMLocator;
class DOMError;
class DOMErrorHandler;

class XPathException;
class XPathEvaluator;
class XPathExpression;
class XPathNSResolver;
class XPathResult;
class XPathSetIterator;
class XPathSetSnapshot;
class XPathNamespace;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::dom

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

