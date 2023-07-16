// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* XmlNode.cc                                                  (C) 2000-2023 */
/*                                                                           */
/* Noeud quelconque d'un arbre DOM.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/XmlNode.h"

#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/XmlException.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/XmlNodeIterator.h"
#include "arcane/core/DomUtils.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XmlNodeSameName
{
 public:
  XmlNodeSameName(const String& name) : m_name(name) {}
 public:
  bool operator()(const XmlNode& node)
    { return node.name()==m_name; }
 private:
  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
_nullNode() const
{
  return XmlNode(m_rm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
child(const String& child_name) const
{
  if (m_node._null())
    return _nullNode();
  XmlNodeSameName same_name(child_name);
  XmlNodeConstIterator i = ARCANE_STD::find_if(begin(),end(),same_name);
  if (i!=end())
    return *i;
  return XmlNode(m_rm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
expectedChild(const String& child_name) const
{
  if (m_node._null())
    return _nullNode();
  XmlNode c = child(child_name);
  if (c.null())
    ARCANE_FATAL("Can not find a child named '{0}' for node '{1}'",child_name,xpathFullName());
  return c;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNodeList XmlNode::
children(const String& child_name) const
{
  XmlNodeList nodes;
  if (m_node._null())
    return nodes;
  XmlNodeSameName same_name(child_name);
  for( XmlNodeConstIterator n = begin(); n!=end(); ++n )
    if (same_name(*n))
      nodes.add(*n);
  return nodes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNodeList XmlNode::
children() const
{
  XmlNodeList nodes;
  if (m_node._null())
    return nodes;
  for( XmlNodeConstIterator n = begin(); n!=end(); ++n )
    nodes.add(*n);
  return nodes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode::eType XmlNode::
type() const
{
  return static_cast<eType>(m_node.nodeType());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String XmlNode::
name() const
{
  if (m_node._null())
    return String();
  return String(m_node.nodeName());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String XmlNode::
xpathFullName() const
{
  StringBuilder full_name;
  if (m_node._null())
    return full_name;

  XmlNode p = parent();
  if (!p.null()){
    full_name = p.xpathFullName();
    full_name.append("/");

    if (m_node.nodeType()==dom::Node::ATTRIBUTE_NODE){
      full_name += "@";
      full_name += name();
    }
    else if (m_node.nodeType()==dom::Node::ELEMENT_NODE){
      full_name.append(name());
      Integer nb_occurence = 1;
      for( XmlNode i=p.front(); i!=(*this); ++i )
        if (i.isNamed(name()))
          ++nb_occurence;
      if (nb_occurence>1){
        full_name += "[";
        full_name += nb_occurence;
        full_name += "]";
      }
    }
    else
      full_name += "?";
  }
  else{
    if (m_node.nodeType()==dom::Node::ATTRIBUTE_NODE){
      full_name = ownerElement().xpathFullName();
      full_name += "/@";
      full_name += name();
    }
    else{
      full_name = String("/");
    }
  }
  return full_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool XmlNode::
isNamed(const String& name) const
{
  return m_node.nodeName()==name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String XmlNode::
value() const
{
  if (null())
    return String();
  return _value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlNode::
setValue(const String& v)
{
  if (m_node._null())
    return;
  if (m_node.nodeType()==dom::Node::ELEMENT_NODE){
    domutils::textContent(m_node,v);
    return;
  }
  m_node.nodeValue(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String XmlNode::
attrValue(const String& name,bool throw_exception) const
{
  String s = domutils::attrValue(m_node,name);
  if (s.null() && throw_exception){
    ARCANE_THROW(XmlException,"No attribute named '{0}' child of '{1}'",
                 name,xpathFullName());
  }
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlNode::
setAttrValue(const String& name,const String& value)
{
  domutils::setAttr(m_node,name,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
attr(const String& name,bool throw_exception) const
{
  dom::Element elem(m_node);
  if (elem._null())
    return _nullNode();

  XmlNode attr_node(m_rm,elem.getAttributeNode(name));
  if (throw_exception && attr_node.null()){
    ARCANE_THROW(XmlException,"No attribute named '{0}' child of '{1}'",
                 name,xpathFullName());
  }
  return attr_node;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
forceAttr(const String& name)
{
  dom::Element elem(m_node);
  if (elem._null())
    return _nullNode();
  dom::Attr attr = elem.getAttributeNode(name);
  if (attr._null()){
    attr = elem.ownerDocument().createAttribute(name);
    attr = elem.setAttributeNode(attr);
  }
  return XmlNode(m_rm,attr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlNode::
removeAttr(const String& name) const
{
  dom::Element elem(m_node);
  elem.removeAttribute(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlNode::
clear()
{
  // Supprime les noeuds fils.
  XmlNode n = front();
  while (!n.null()){
    remove(n);
    n = front();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlNode::
remove()
{
  if (m_node._null())
    return;
  dom::Node parent = m_node.parentNode();
  if (parent._null())
    return;
  parent.removeChild(m_node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlNode::
remove(const XmlNode& child_node)
{
  dom::Node child = m_node.removeChild(child_node.domNode());
  child.releaseNode();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
nextWithName(const String& name) const
{
  //\todo a tester
  if (m_node._null())
    return _nullNode();
  XmlNode n(next());
  while(!n.null() && !n.isNamed(name))
    ++n;
  return n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
prevWithName(const String& name) const
{
  //\todo a tester
  if (m_node._null())
    return _nullNode();
  XmlNode n(prev());
  while(!n.null() && !n.isNamed(name))
    --n;
  return n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
nextSameType() const
{
  //\todo a tester
  if (m_node._null())
    return _nullNode();
  XmlNode n(next());
  while(!n.null() && n.type()!=type())
    ++n;
  return n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
prevSameType() const
{
  //\todo a tester
  if (m_node._null())
    return _nullNode();
  XmlNode n(prev());
  while(!n.null() && n.type()!=type())
    --n;
  return n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlNode::
replace(const XmlNode& new_node,XmlNode& ref_node)
{
  if (m_node._null() || new_node.null() || ref_node.null())
    return;
  m_node.replaceChild(new_node.domNode(),ref_node.domNode());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlNode::
_throwBadConvert(const char* type_name,const String& value) const
{
  ARCANE_THROW(XmlException,"XML Node '{0}' can not convert value '{1}' to type '{2}'",
               xpathFullName(),value,type_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool XmlNode::
valueAsBoolean(bool throw_exception) const
{
  if (null())
    return false;
  String value = _value();
  if (value=="false" || value=="0")
    return false;
  if (value=="true" || value=="1")
    return true;
  if (throw_exception)
    ARCANE_THROW(XmlException,"XML Node '{0}' can not convert value '{1}' to type 'bool'."
                 " Valid values are 'true', 'false', '0' (zero) ou '1'.",
                 xpathFullName(),value);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer XmlNode::
valueAsInteger(bool throw_exception) const
{
  if (null())
    return 0;
  String value = _value();
  Integer v = 0;
  if (builtInGetValue(v,value))
    if (throw_exception)
      _throwBadConvert("Integer",value);
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 XmlNode::
valueAsInt64(bool throw_exception) const
{
  if (null())
    return 0;
  String value = _value();
  Int64 v = 0;
  if (builtInGetValue(v,value))
    if (throw_exception)
      _throwBadConvert("Int64",value);
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real XmlNode::
valueAsReal(bool throw_exception) const
{
  if (null())
    return 0.0;
  String value = _value();
  Real v = 0.0;
  if (builtInGetValue(v,value))
    if (throw_exception)
      _throwBadConvert("Real",value);
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
childWithAttr(const String& elem_name,const String& attr_name,
              const String& attr_value) const
{
  String name;
  if (null())
    return _nullNode();
  for( XmlNode::const_iter i(*this); i(); ++i ){
    if (!i->isNamed(elem_name))
      continue;
    name = i->attrValue(attr_name);
    if (name==attr_value)
      return *i;
  }
  return XmlNode(m_rm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
childWithNameAttr(const String& elem_name,const String& attr_value) const
{
  return childWithAttr(elem_name,String("name"),attr_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
xpathNode(const String& xpath_expr) const
{
  return XmlNode(m_rm,domutils::nodeFromXPath(m_node,xpath_expr));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String XmlNode::
_value() const
{
  if (m_node.nodeType()==dom::Node::ELEMENT_NODE)
    return domutils::textContent(m_node);
  return String(m_node.nodeValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlNode::
assignDomNode(const dom::Node& node)
{
  m_node = node;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
insertAfter(const XmlNode& new_child,const XmlNode& ref_node)
{
  if (new_child.null())
    return _nullNode();
  XmlNode next = ref_node;
  if (!next.null())
    ++next;
  if (next.null())
    append(new_child);
  else
    m_node.insertBefore(new_child.domNode(),next.domNode());
  return new_child;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
documentElement() const
{
  dom::Document doc(m_node);
  if (doc._null())
    return _nullNode();
  return _build(doc.documentElement());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
ownerElement() const
{
  dom::Attr attr(m_node);
  if (attr._null())
    return _nullNode();
  return _build(attr.ownerElement());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
createElement(const String& name)
{
  return createNode(ELEMENT,name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
createAndAppendElement(const String& name,const String& value)
{
  XmlNode n = createNode(ELEMENT,name,value);
  append(n);
  return n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
createAndAppendElement(const String& name)
{
  XmlNode n = createNode(ELEMENT,name);
  append(n);
  return n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
createNode(eType type,const String& name_or_value)
{
  dom::Document doc(m_node);
  if (doc._null())
    doc = m_node.ownerDocument();
  XmlNode ret_node(m_rm);
  String nov = name_or_value;
  switch(type){
   case ELEMENT:
     ret_node.assignDomNode(doc.createElement(nov));
     break;
   case TEXT:
     ret_node.assignDomNode(doc.createTextNode(nov));
     break;
   default:
     ARCANE_THROW(NotImplementedException,
                  "createNode() not implemented for node type {0}",(int)type);
  }
  return ret_node;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
createText(const String& value)
{
  return createNode(TEXT,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
createNode(eType type,const String& name,const String& value)
{
  dom::Document doc(m_node);
  if (doc._null())
    doc = m_node.ownerDocument();
  XmlNode ret_node(m_rm);
  switch(type){
   case ELEMENT:
     ret_node.assignDomNode(doc.createElement(name));
     ret_node.setValue(value);
     break;
   case TEXT:
     ret_node.assignDomNode(doc.createTextNode(value));
     break;
   default:
     ARCANE_THROW(NotImplementedException,
                  "createNode() not implemented for node type {0}",(int)type);
  }
  return ret_node;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode XmlNode::
_build(const dom::Node& node) const
{
  return XmlNode(m_rm,node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlElement::
XmlElement(XmlNode& parent,const String& name,const String& value)
: XmlNode(parent)
{
  _setNode(parent.createAndAppendElement(name,value).domNode());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlElement::
XmlElement(XmlNode& parent,const String& name)
: XmlNode(parent)
{
  _setNode(parent.createAndAppendElement(name).domNode());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNodeNameIterator::
XmlNodeNameIterator(const XmlNode& from,const String& ref_name)
: m_parent   (from)
, m_current  (0)
, m_ref_name (ref_name)
{
  _findNextValid(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNodeNameIterator::
XmlNodeNameIterator(const XmlNode& from,const char* ref_name)
: m_parent   (from)
, m_current  (0)
, m_ref_name (String(ref_name))
{
  _findNextValid(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlNodeNameIterator::
_findNextValid(bool is_init)
{
  if (is_init)
    m_current = m_parent.front();
  else{
    if (m_current.null())
      return;
    ++m_current;
  }
  while (!m_current.null()){
    if (m_current.isNamed(m_ref_name))
      break;
    ++m_current;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

