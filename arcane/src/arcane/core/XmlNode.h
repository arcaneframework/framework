// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* XmlNode.h                                                   (C) 2000-2023 */
/*                                                                           */
/* Any node in a DOM tree.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_XMLNODE_H
#define ARCANE_CORE_XMLNODE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/core/Dom.h"

#include <iterator>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IRessourceMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XmlNodeIterator;
class XmlNodeConstIterator;
class XmlNodeList;

/*!
 * \ingroup Xml
 * \brief Node of a DOM tree.
 *
 * This class is used for all types of DOM nodes and allows
 * them to be manipulated more simply than with the DOM without having
 * to perform type conversions.
 *
 * Each node can be considered a container in the sense of the STL.
 */
class ARCANE_CORE_EXPORT XmlNode
{
 public:

  //! Type of the elements in the array
  typedef XmlNode value_type;
  //! Type of the iterator over an element in the array
  typedef XmlNodeIterator iterator;
  //! Type of the constant iterator over an element in the array
  typedef XmlNodeConstIterator const_iterator;
  //! Type pointer of an element in the array
  typedef value_type* pointer;
  //! Type constant pointer of an element in the array
  typedef const value_type* const_pointer;
  //! Type reference of an element in the array
  typedef value_type& reference;
  //! Type constant reference of an element in the array
  typedef const value_type& const_reference;
  //! Type indexing the array
  typedef Integer size_type;
  //! Type of a distance between iterator elements in the array
  typedef int difference_type;

  //! Type of an iterator over the entire array
  typedef IterT<XmlNode> iter;
  //! Type of a constant iterator over the entire array
  typedef ConstIterT<XmlNode> const_iter;

 public:
  
  /*! \brief NodeType
    An integer indicating which type of node this is.
    \note Numeric codes up to 200 are reserved to W3C for possible future use.
  */
  enum eType
  {
    //! The node is an Element
    ELEMENT = 1,
    //! The node is an Attr
    ATTRIBUTE = 2,
    //! The node is a Text node
    TEXT = 3,
    //! The node is a CDATASection
    CDATA_SECTION = 4,
    //! The node is an EntityReference
    ENTITY_REFERENCE = 5,
    //! The node is an Entity
    ENTITY = 6,
    //! The node is a ProcessingInstruction
    PROCESSING_INSTRUCTION = 7,
    //! The node is a Comment
    COMMENT = 8,
    //! The node is a Document
    DOCUMENT = 9,
    //! The node is a DocumentType
    DOCUMENT_TYPE = 10,
    //! The node is a DocumentFragment
    DOCUMENT_FRAGMENT = 11,
    //! The node is a Notation
    NOTATION = 12
  };

 public:

  XmlNode(IRessourceMng* m,const dom::Node& node) : m_rm(m), m_node(node) {}
  //TODO: to be removed
  explicit XmlNode(IRessourceMng* m) : m_rm(m), m_node() {}
  XmlNode() : m_rm(nullptr), m_node() {}

 public:

  //! Returns an iterator over the first element of the array
  inline iterator begin();
  //! Returns an iterator over the first element after the end of the array
  inline iterator end();
  //! Returns a constant iterator over the first element of the array
  inline const_iterator begin() const;
  //! Returns a constant iterator over the first element after the end of the array
  inline const_iterator end()   const;

 public:

  //! Node type
  eType type() const;

  //! Node name
  String name() const;

  /*! \brief XPath name of the node with its ancestors.
   * \warning Only works for elements.
   */  
  String xpathFullName() const;

  //! True if the element name is \a name
  bool isNamed(const String& name) const;

  /*! \brief Node value.
   *
   * For an element, it is the concatenation of the values of each
   * of its child nodes of type TEXT, CDATA_SECTION or ENTITY_REFERENCE.
   * For nodes other than elements, it is the value of
   * the Node::nodeValue() method of the DOM.
   */
  String value() const;

  /*! \brief Node value converted to integer.
   *
   * If conversion fails, if \a throw_exception
   * is \a false returns 0, otherwise throws an exception.
   */
  Integer valueAsInteger(bool throw_exception=false) const;

  /*! \brief Node value converted to 64-bit integer. 0 if conversion fails.
   *
   * If conversion fails, if \a throw_exception
   * is \a false returns 0, otherwise throws an exception.
   */
  Int64 valueAsInt64(bool throw_exception=false) const;

  /*! \brief Node value converted to boolean.
   *
   * A value of \c false or \c 0 corresponds to \a false. A value
   * of \c true or \c 1 corresponds to \a true.
   * If conversion fails, if \a throw_exception
   * is \a false returns \a false, otherwise throws an exception.
   */
  bool valueAsBoolean(bool throw_exception=false) const;

  /*! \brief Node value converted to real number.
   * If conversion fails, if \a throw_exception
   * is \a false returns 0.0, otherwise throws an exception.
   */
  Real valueAsReal(bool throw_exception=false) const;

  /*! \brief Sets the node value.
   *
   * This method is only valid for ELEMENT_NODE or
   * ATTRIBUTE_NODE. For elements, it deletes all children
   * and adds a single TEXT_NODE child containing the value \a value
   */
  void setValue(const String& value);

  /*! \brief Value of attribute \a name.
   *
   * If the attribute does not exist, if \a throw_exception is \a false returns
   * the null string, otherwise throws an exception.
   */
  String attrValue(const String& name,bool throw_exception=false) const;

  //! Sets the attribute \a name to the value \a value
  void setAttrValue(const String& name,const String& value);

  /*!
   * \brief Returns the attribute of name \a name.
   *
   * If the attribute does not exist, if \a throw_exception is \a false returns
   * a null node, otherwise throws an exception.
   */
  XmlNode attr(const String& name,bool throw_exception=false) const;

  /*!
   * \brief Returns the attribute of name \a name.
   * If no attribute with this name exists, an attribute with
   * the null string as value is created and returned.
   */
  XmlNode forceAttr(const String& name);

  /*!
   * \brief Removes the attribute of name \a name from this node.
   * If this node is not an element, nothing is done.
   */
  void removeAttr(const String& name) const;

  /*!
   * \brief Returns the document element.
   * \pre type()==DOCUMENT_NODE
   */
  XmlNode documentElement() const;

  /*!
   * \brief Returns the owning element of this attribute.
   * \pre type()==ATTRIBUTE_NODE
   */
  XmlNode ownerElement() const;

  //! Deletes all child nodes
  void clear();

  /*!
   * \brief Child node of this node with name \a name
   *
   * If multiple nodes with this name exist, returns the first one.
   * If the node is not found, returns a null node
   */
  XmlNode child(const String& name) const;

  /*!
   * \brief Child node of this node with name \a name
   *
   * If multiple nodes with this name exist, returns the first one.
   * If the node is not found, throws an exception.
   */
  XmlNode expectedChild(const String& name) const;

  //! Set of child nodes of this node having the name \a name
  XmlNodeList children(const String& name) const;

  //! Set of child nodes of this node
  XmlNodeList children() const;

  //! Parent of this node (null if none)
  XmlNode parent() const { return XmlNode(m_rm,m_node.parentNode()); }

  /*! \brief Adds \a child_node as a child of this node.
   *
   * The node is added after all children.
   */
  void append(const XmlNode& child_node) { m_node.appendChild(child_node.domNode()); }
  //! Removes the child node \a child_node
  void remove(const XmlNode& child_node);
  //! Replaces the child node \a ref_node with the node \a new_node
  void replace(const XmlNode& new_node,XmlNode& ref_node);
  //! Removes this node from the document
  void remove();
  //! First child
  XmlNode front() const { return XmlNode(m_rm,m_node.firstChild()); }
  //! Last child
  XmlNode last() const { return XmlNode(m_rm,m_node.lastChild()); }
  //! Next node (nextSibling())
  XmlNode next() const { return XmlNode(m_rm,m_node.nextSibling()); }
  //! Previous node (previousSibling())
  XmlNode prev() const { return XmlNode(m_rm,m_node.previousSibling()); }
  //! Returns the next node after this node having the name \a name.
  XmlNode nextWithName(const String& name) const;
  //! Returns the previous node before this node having the name \a name.
  XmlNode prevWithName(const String& name) const;
  //! Returns the next node of the same type.
  XmlNode nextSameType() const;
  //! Returns the previous node of the same type.
  XmlNode prevSameType() const;
  void operator++() { m_node = m_node.nextSibling(); }
  void operator--() { m_node = m_node.previousSibling(); }

  //! True if the node is null
  bool null() const { return m_node._null(); }
  bool operator!() const { return null(); }

  //! \internal
  dom::Node domNode() const { return m_node; }
  //! \internal
  void assignDomNode(const dom::Node& node);

  /*!
   * \brief Inserts a node.
   * Inserts the node \a new_child after the node \a ref_node.
   * If \a new_child is \c null, does nothing.
   * If \a ref_node is \c null, \a new_child is added to the end (like append()). Otherwise,
   * \a ref_node must be a child of this node and \a new_child is inserted after
   * \a ref_node.
   * On success, returns the added node (\a new_child), otherwise the null node.
   */
  XmlNode insertAfter(const XmlNode& new_child,const XmlNode& ref_node);

  /*!
   * \brief Returns the child of this node having the name \a elem_name and
   * an attribute of name \a attr_name with value \a attr_value.
   */
  XmlNode childWithAttr(const String& elem_name,const String& attr_name,
			const String& attr_value) const;
  /*!
   * \brief Returns the child of this node having the name \a elem_name and
   * an attribute of name \c "name" with value \a attr_value.
   */
  XmlNode childWithNameAttr(const String& elem_name,
			    const String& attr_value) const;

  /*!
   * \brief Returns a node from an XPath expression.
   * \param xpath_expr XPath expression.
   */
  XmlNode xpathNode(const String& xpath_expr) const;

  /*!
   * \brief Creates a node of a given type.
   *
   * If type() is not DOCUMENT_NODE, it uses ownerDocument() as
   * factory.
   *
   * \param type type of the node.
   * \param name of the node.
   * \param value of the node.
   * \return the created node.
   * \pre type()==DOCUMENT_NODE
   */
  XmlNode createNode(eType type,const String& name,const String& value);

  /*!
   * \brief Creates a node of a given type.
   *
   * If type() is not DOCUMENT_NODE, it uses ownerDocument() as
   * factory.
   *
   * \param type type of the node.
   * \param name or value of the node in the case where the node has no name.
   * \return the created node.
   */
  XmlNode createNode(eType type,const String& name_or_value);

  /*!
   * \brief Creates a text node.
   * \param value value of the text node.
   * \return the created node.
   */
  XmlNode createText(const String& value);

  XmlNode createElement(const String& name);

  XmlNode createAndAppendElement(const String& name);

  XmlNode createAndAppendElement(const String& name,const String& value);

  XmlNode ownerDocument() const { return XmlNode(m_rm,m_node.ownerDocument()); }

  IRessourceMng* rm() const { return m_rm; }

 private:

  IRessourceMng* m_rm;
  dom::Node m_node;
  
 protected:

  String _value() const;
  XmlNode _build(const dom::Node& node) const;
  XmlNode _nullNode() const;
  void _setNode(const dom::Node& n) { m_node = n; }
  inline void _throwBadConvert(const char* type_name,const String& value) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Element of a DOM tree.
 */
class ARCANE_CORE_EXPORT XmlElement
: public XmlNode
{
 public:
  /*! \brief Creates a child element of \a parent.
   * The created element has the name \a name and the value \a value. 
   * It is added to the end of the list of children of \a parent.
   */
  XmlElement(XmlNode& parent,const String& name,const String& value);
  /*! \brief Creates a child element of \a parent.
   * The created element has the name \a name and the value \a value.
   * It is added to the end of the list of children of \a parent.
   */
  XmlElement(XmlNode& parent,const String& name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
operator==(const XmlNode& n1,const XmlNode& n2)
{
  return n1.domNode()==n2.domNode();
}

inline bool
operator!=(const XmlNode& n1,const XmlNode& n2)
{
  return n1.domNode()!=n2.domNode();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
