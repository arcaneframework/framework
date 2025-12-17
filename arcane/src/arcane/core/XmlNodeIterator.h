// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* XmlNodeIterator.h                                           (C) 2000-2025 */
/*                                                                           */
/* Iterator sur les noeuds d'un arbre DOM.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_XMLNODEITERATOR_H
#define ARCANE_CORE_XMLNODEITERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/XmlNode.h"

#include <iterator>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XmlNodeConstIterator
{
 public:

  typedef std::bidirectional_iterator_tag iterator_category;
  typedef XmlNode value_type;
  typedef int difference_type;
  typedef XmlNode* pointer;
  typedef XmlNode& reference;

 public:

  XmlNodeConstIterator(const XmlNode& node)
  : m_node(node)
  {}
  XmlNodeConstIterator()
  : m_node()
  {}

 public:

  void operator++() { ++m_node; }
  void operator--() { --m_node; }
  const XmlNode& operator*() const { return m_node; }
  const XmlNode* operator->() const { return &m_node; }

 protected:

  XmlNode m_node;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XmlNodeIterator
: public XmlNodeConstIterator
{
 public:

  XmlNodeIterator(const XmlNode& node)
  : XmlNodeConstIterator(node)
  {}
  XmlNodeIterator() {}

 public:

  const XmlNode& operator*() const { return m_node; }
  const XmlNode* operator->() const { return &m_node; }
  XmlNode& operator*() { return m_node; }
  XmlNode* operator->() { return &m_node; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
operator==(const XmlNodeConstIterator& n1, const XmlNodeConstIterator& n2)
{
  return *n1 == *n2;
}

inline bool
operator!=(const XmlNodeConstIterator& n1, const XmlNodeConstIterator& n2)
{
  return *n1 != *n2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline XmlNodeIterator XmlNode::
begin()
{
  return XmlNode(m_rm, m_node.firstChild());
}

inline XmlNodeIterator XmlNode::
end()
{
  return XmlNode(m_rm);
}

inline XmlNodeConstIterator XmlNode::
begin() const
{
  return XmlNode(m_rm, m_node.firstChild());
}

inline XmlNodeConstIterator XmlNode::
end() const
{
  return XmlNode(m_rm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Itérateur sur les fils d'un noeud \a from de nom \a ref_name.
 */
class XmlNodeNameIterator
{
 public:

  XmlNodeNameIterator(const XmlNode& from, const String& ref_name);
  XmlNodeNameIterator(const XmlNode& from, const char* ref_name);
  bool operator()() const { return !m_current.null(); }
  void operator++() { _findNextValid(false); }
  const XmlNode& operator*() const { return m_current; }
  const XmlNode* operator->() const { return &m_current; }
  XmlNode& operator*() { return m_current; }
  XmlNode* operator->() { return &m_current; }

 private:

  XmlNode m_parent;
  XmlNode m_current;
  String m_ref_name;

 private:

  void _findNextValid(bool is_init);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

