// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DomUtils.cc                                                 (C) 2000-2018 */
/*                                                                           */
/* Fonctions utilitaires diveres concernant le DOM.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/String.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/Dom.h"
#include "arcane/DomUtils.h"
#include "arcane/XmlNode.h"
#include "arcane/IXmlDocumentHolder.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

ARCANE_BEGIN_NAMESPACE_DOMUTILS

extern "C++" ARCANE_CORE_EXPORT void
removeAllChildren(const dom::Node& parent);

extern "C++" ARCANE_CORE_EXPORT bool
writeNode(std::ostream& ostr,const dom::Node&);

extern "C++" ARCANE_CORE_EXPORT bool
writeNodeChildren(std::ostream& ostr,const dom::Node&);

ARCANE_END_NAMESPACE_DOMUTILS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
_notImplemented(const char* reason)
{
  cerr << "* DOMUTILS NOT YET IMPLEMENTED: " << reason << '\n';
  throw dom::DOMException(dom::NOT_IMPLEMENTED_ERR);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
_writeNodeChildren(std::ostream& o,const dom::Node& node)
{
  // Affiche récursivement les noeuds fils
  dom::Node next = node.firstChild();
  while (!next._null()){
    domutils::writeNode(o,next);
    next = next.nextSibling();
  }
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String domutils::
textContent(const dom::Node& node)
{
  using namespace dom;
  StringBuilder str;
  if (node._null())
    return str.toString();

  if (node.nodeType()!=Node::ELEMENT_NODE)
    ARCANE_THROW(NotImplementedException,"get text value for non ELEMENT_NODE");
  for( Node i=node.firstChild(); !i._null(); i=i.nextSibling() ){
    UShort ntype = i.nodeType();
    if (ntype==Node::TEXT_NODE)
      str += i.nodeValue();
    else if (ntype==Node::CDATA_SECTION_NODE){
      str += i.nodeValue();
    }
    else if (ntype==Node::ENTITY_REFERENCE_NODE){
      ARCANE_THROW(NotImplementedException,"get text value for non ENTITY_REFERENCE_NODE");
    }
  }
  return str.toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void domutils::
textContent(dom::Node& node,const String& new_value)
{
  // Même sémantique que Node::textContent du DOM3:
  // - Supprime tous les noeuds fils.
  // - création d'un seul noeud texte contenant la nouvelle valeur
  using namespace dom;
  if (node.nodeType()!=Node::ELEMENT_NODE)
    ARCANE_THROW(NotImplementedException,"set text value for non ELEMENT_NODE");
  removeAllChildren(node);
  if (!new_value.null()){
    Text text_node = node.ownerDocument().createTextNode(new_value);
    node.appendChild(text_node);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String domutils::
textValue(const dom::Node& node)
{
  return domutils::textContent(node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void domutils::
textValue(dom::Node& node,const String& new_value)
{
  textContent(node,new_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

dom::Element domutils::
createElement(const dom::Node& parent,const String& name,const String& value)
{
  if (parent._null())
    return dom::Element();
  dom::Document doc = parent.ownerDocument();
  if (doc._null())
    return dom::Element();
  dom::Element elem = doc.createElement(name);
  textContent(elem,value);
  parent.appendChild(elem);
  return elem;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String domutils::
attrValue(const dom::Node& node,const String& attr_name)
{
  // TODO: Utiliser directement les méthodes du DOM que sont getAttribute()
  // mais il faut pour cela gérer les namespace.
  String str;
  if (node._null())
    return str;
  const dom::NamedNodeMap& attr = node.attributes();
  if (attr._null())
    return str;
  const dom::Node& n = attr.getNamedItem(attr_name);
  if (n._null())
    return str;
  str = n.nodeValue();

#if 0
  // A activer lorsque l'implémentation via getAttribute() sera effective.
  {
    dom::Element element{node};
    String str2 = element.getAttribute(attr_name);
    if (str2!=str)
      ARCANE_FATAL("Bad new value for attribute '{0}' new={1} current={2}",attr_name,str2,str);
  }
#endif

  return str;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void domutils::
setAttr(const dom::Element& elem,const String& name,const String& value)
{
  if (elem._null())
    return;
  elem.setAttribute(name,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

dom::Node domutils::
childNode(const dom::Node& parent,const String& child_name)
{
  dom::DOMString ref_name(child_name);
  for( dom::Node i=parent.firstChild(); !i._null(); i=i.nextSibling() ){
    if (i.nodeName()==ref_name)
      return i;
  }
  return dom::Node();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void domutils::
removeAllChildren(const dom::Node& parent)
{
  if (parent._null())
    return;

  // Supprime les noeuds fils.
  dom::Node n = parent.firstChild();
  while (!n._null()){
    parent.removeChild(n);
    // TODO LIBERER LA MEMOIRE
    n = parent.firstChild();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne le noeud correspondant à une expression XPath.
 * Retourne le noeud correspondant à l'expression \a xpath_expr avec
 * pour contexte le noeud \a context_node.
 * L'implémentation actuelle ne supporte que le type suivant d'expression:
 * - a/b*.
 */
dom::Node domutils::
nodeFromXPath(const dom::Node& context_node,const String& xpath_expr)
{
  const char* expr = xpath_expr.localstr();
  if (context_node._null()){
    return dom::Node();
  }
  if (!expr){
    return dom::Node();
  }
  const char* separator = ::strchr(expr,'/');
  if (separator){
    // Chaîne de type \a a/b. Recherche le noeud fils de nom \a a et
    // lui applique récursivement cette fonction avec \a b comme expression.
    std::string_view buf1(expr,(Int64)(separator-expr));
    String buf(buf1);
    dom::Node child = childNode(context_node,buf);
    if (child._null()){
      return dom::Node();
    }
    return nodeFromXPath(child,String(std::string_view(separator+1)));
  }
  return childNode(context_node,xpath_expr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool domutils::
writeNodeChildren(std::ostream& ostr,const dom::Node& node)
{
  _writeNodeChildren(ostr,node);
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool domutils::
writeNode(std::ostream& o,const dom::Node& node)
{
  using namespace dom;
  switch(node.nodeType()){
   case Node::ELEMENT_NODE:
     {
       o << '<' << node.nodeName();
       NamedNodeMap attr_list = node.attributes();
       for( ULong i=0, s=attr_list.length(); i<s; ++i ){
         o << ' ';
         writeNode(o,attr_list.item(i));
       }
       o << '>';
       writeNodeChildren(o,node);
       o << "</" << node.nodeName() << '>';
     }
     break;
   case Node::ATTRIBUTE_NODE:
     o << node.nodeName() << '='
       << '"' << node.nodeValue() << '"';
     break;
   case Node::TEXT_NODE:
     o << node.nodeValue();
     //_notImplemented("Dom::writeNode() for TEXT_NODE");
     break;
   case Node::CDATA_SECTION_NODE:
     o << node.nodeValue();
     cerr << "** Dom::writeNode() for CDATA_SECTION_NODE Not fully implemented\n";
     //_notImplemented("Dom::writeNode() for CDATA_SECTION_NODE");
     break;
   case Node::ENTITY_REFERENCE_NODE:
     _notImplemented("Dom::writeNode() for ENTITY_REFERENCE_NODE");
     break;
   case Node::ENTITY_NODE:
     _notImplemented("Dom::writeNode() for ENTITY_NODE");
     break;
   case Node::PROCESSING_INSTRUCTION_NODE:
     _notImplemented("Dom::writeNode() for PROCESSING_INSTRUCTION_NODE");
     break;
   case Node::COMMENT_NODE:
     o << "<!--" << node.nodeValue() << "-->";
     break;
   case Node::DOCUMENT_NODE:
     _writeNodeChildren(o,node);
     break;
   case Node::DOCUMENT_TYPE_NODE:
     _writeNodeChildren(o,node);
     break;
   case Node::DOCUMENT_FRAGMENT_NODE:
     _notImplemented("Dom::writeNode() for DOCUMENT_FRAGMENT_NODE");
     break;
   case Node::NOTATION_NODE:
     _notImplemented("Dom::writeNode() for NOTATION_NODE");
     break;
   default:
     _notImplemented("Dom::writeNode() for unknown node");
     break;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool domutils::
saveDocument(std::ostream& ostr,const dom::Document& doc,int indent_level)
{
  ByteUniqueArray bytes;
  saveDocument(bytes,doc,indent_level);
  ostr.write((const char*)bytes.data(),bytes.size());
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool domutils::
saveDocument(ByteArray& bytes,const dom::Document& doc,int indent_level)
{
  dom::DOMImplementation domimp;
  domimp._save(bytes,doc,indent_level);
  Integer nb_byte = bytes.size();
  if (nb_byte>=1 && bytes[nb_byte-1]=='\0'){
    ARCANE_FATAL("Invalid null charactere at end of XML stream");
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* domutils::
createXmlDocument()
{
  dom::DOMImplementation domimp;
  return domimp._newDocument();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

domutils::NameIterator::
NameIterator(const dom::Node& from,const String& ref_name)
: m_parent(from)
, m_current()
, m_ref_name(ref_name)
{
  _findNextValid(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void domutils::NameIterator::
_findNextValid(bool is_init)
{
  if (is_init)
    m_current = m_parent.firstChild();
  else{
    if (m_current._null())
      return;
    m_current = m_current.nextSibling();
  }
  while (!m_current._null()){
    if (m_current.nodeName()==m_ref_name)
      break;
    m_current = m_current.nextSibling();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* IXmlDocumentHolder::
loadFromBuffer(Span<const Byte> buffer,const String& name,ITraceMng* tm)
{
  dom::DOMImplementation domimp;
  // Lecture du fichier contenant les informations internes.
  return domimp._load(asBytes(buffer),name,tm);
}

IXmlDocumentHolder* IXmlDocumentHolder::
loadFromBuffer(ByteConstSpan buffer,const String& name,ITraceMng* tm)
{
  dom::DOMImplementation domimp;
  // Lecture du fichier contenant les informations internes.
  return domimp._load(buffer,name,tm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* IXmlDocumentHolder::
loadFromFile(const String& filename,ITraceMng* tm)
{
  return loadFromFile(filename,String(),tm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* IXmlDocumentHolder::
loadFromFile(const String& filename,const String& schema_filename,ITraceMng* tm)
{
  dom::DOMImplementation domimp;
  return domimp._load(filename,tm,schema_filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

