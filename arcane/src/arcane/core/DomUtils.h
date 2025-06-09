// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DomUtils.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires divers concernant le DOM.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_DOMUTILS_H
#define ARCANE_CORE_DOMUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Dom.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Les méthodes de ce fichier sont internes à Arcane et ne doivent
 * pas être utilisées ailleurs.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::domutils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Si le noeud est un élément, retourne la concaténation de ses fils textes,
// sinon, retourn nodeValue().
// Utiliser textContext() à la place
extern "C++" ARCANE_DEPRECATED_260 ARCANE_CORE_EXPORT String
textValue(const dom::Node& node);

// Utiliser textContext() à la place
extern "C++" ARCANE_DEPRECATED_260 ARCANE_CORE_EXPORT void
textValue(dom::Node& node, const String& new_value);

extern "C++" ARCANE_CORE_EXPORT String
textContent(const dom::Node& node);

extern "C++" ARCANE_CORE_EXPORT void
textContent(dom::Node& node, const String& new_value);

extern "C++" ARCANE_CORE_EXPORT dom::Element
createElement(const dom::Node& parent, const String& name, const String& value);

extern "C++" ARCANE_CORE_EXPORT String
attrValue(const dom::Node& node, const String& attr_name);

extern "C++" ARCANE_CORE_EXPORT void
setAttr(const dom::Element& node, const String& name, const String& value);

extern "C++" ARCANE_CORE_EXPORT dom::Node
childNode(const dom::Node& parent, const String& child_name);

extern "C++" ARCANE_CORE_EXPORT dom::Node
nodeFromXPath(const dom::Node& context_node, const String& xpath_expr);

extern "C++" ARCANE_DEPRECATED_260 ARCANE_CORE_EXPORT bool
saveDocument(std::ostream& istr, const dom::Document&, int indent_level = -1);

extern "C++" ARCANE_CORE_EXPORT bool
saveDocument(ByteArray& bytes, const dom::Document&, int indent_level = -1);

extern "C++" ARCANE_CORE_EXPORT IXmlDocumentHolder*
createXmlDocument();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * Itérateur sur les fils d'un noeud \a from de nom \a ref_name.
 */
class ARCANE_CORE_EXPORT NameIterator
{
 public:

  NameIterator(const dom::Node& from, const String& ref_name);
  //NameIterator(const dom::Node& from,const char* ref_name);
  bool operator()() const { return !m_current._null(); }
  void operator++() { _findNextValid(false); }
  const dom::Node& operator*() const { return m_current; }
  const dom::Node* operator->() const { return &m_current; }

 private:

  dom::Node m_parent;
  dom::Node m_current;
  dom::DOMString m_ref_name;

 private:

  void _findNextValid(bool is_init);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::domutils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

