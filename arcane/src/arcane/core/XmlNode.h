// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* XmlNode.h                                                   (C) 2000-2023 */
/*                                                                           */
/* Noeud quelconque d'un arbre DOM.                                          */
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
 * \brief Noeud d'un arbre DOM.
 *
 * Cette classe sert pour tous les types de noeuds du DOM et permet
 * de les manipuler de manière plus simple qu'avec le DOM sans avoir
 * besoin de faire des conversions de types.
 *
 * Chaque noeud peut être considéré comme un container au sens de la STL.
 */
class ARCANE_CORE_EXPORT XmlNode
{
 public:

  //! Type des éléments du tableau
  typedef XmlNode value_type;
  //! Type de l'itérateur sur un élément du tableau
  typedef XmlNodeIterator iterator;
  //! Type de l'itérateur constant sur un élément du tableau
  typedef XmlNodeConstIterator const_iterator;
  //! Type pointeur d'un élément du tableau
  typedef value_type* pointer;
  //! Type pointeur constant d'un élément du tableau
  typedef const value_type* const_pointer;
  //! Type référence d'un élément du tableau
  typedef value_type& reference;
  //! Type référence constante d'un élément du tableau
  typedef const value_type& const_reference;
  //! Type indexant le tableau
  typedef Integer size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef int difference_type;

  //! Type d'un itérateur sur tout le tableau
  typedef IterT<XmlNode> iter;
  //! Type d'un itérateur constant sur tout le tableau
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
  //TODO: à supprimer
  explicit XmlNode(IRessourceMng* m) : m_rm(m), m_node() {}
  XmlNode() : m_rm(nullptr), m_node() {}

 public:

  //! Retourne un iterateur sur le premier élément du tableau
  inline iterator begin();
  //! Retourne un iterateur sur le premier élément après la fin du tableau
  inline iterator end();
  //! Retourne un iterateur constant sur le premier élément du tableau
  inline const_iterator begin() const;
  //! Retourne un iterateur constant sur le premier élément après la fin du tableau
  inline const_iterator end()   const;

 public:

  //! Type du noeud
  eType type() const;

  //! Nom du noeud
  String name() const;

  /*! \brief Nom XPath du noeud avec ces ancêtres.
   * \warning Ne fonctionne que pour les éléments.
   */  
  String xpathFullName() const;

  //! Vrai si le nom de l'élément est \a name
  bool isNamed(const String& name) const;

  /*! \brief Valeur du noeud.
   *
   * Pour un élément, il s'agit de la concaténation des valeurs de chacun
   * de ses noeuds fils de type TEXT, CDATA_SECTION ou ENTITY_REFERENCE.
   * Pour les noeuds autre que les éléments, il s'agit de la valeur de
   * la méthode Node::nodeValue() du DOM.
   */
  String value() const;

  /*! \brief Valeur du noeud convertie en entier.
   *
   * Si la conversion échoue, si \a throw_exception
   * vaut \a false retourne 0, sinon lève une exception.
   */
  Integer valueAsInteger(bool throw_exception=false) const;

  /*! \brief Valeur du noeud convertie en entier 64 bits. 0 si conversion échoue.
   *
   * Si la conversion échoue, si \a throw_exception
   * vaut \a false retourne 0, sinon lève une exception.
   */
  Int64 valueAsInt64(bool throw_exception=false) const;

  /*! \brief Valeur du noeud convertie en booléan.
   *
   * Une valeur de \c false ou \c 0 correspond à \a false. Une valeur
   * de \c true ou \c 1 correspond à \a true.
   * Si la conversion échoue, si \a throw_exception
   * vaut \a false retourne \a false, sinon lève une exception.
   */
  bool valueAsBoolean(bool throw_exception=false) const;

  /*! \brief Valeur du noeud convertie en réel.
   * Si la conversion échoue, si \a throw_exception
   * vaut \a false retourne 0.0, sinon lève une exception.
   */
  Real valueAsReal(bool throw_exception=false) const;

  /*! \brief Positionne la valeur du noeud.
   *
   * Cette méthode n'est valide que pour les noeuds ELEMENT_NODE ou
   * ATTRIBUTE_NODE. Pour les éléments, elles supprime tous les fils
   * et ajoute un seul fils de type TEXT_NODE comportant la valeur \a value
   */
  void setValue(const String& value);

  /*! \brief Valeur de l'attribut \a name.
   *
   * Si l'attribut n'existe pas, si \a throw_exception vaut \a false retourne
   * la chaîne nulle, sinon lève une exception.
   */
  String attrValue(const String& name,bool throw_exception=false) const;

  //! Positionne l'attribut \a name à la valeur \a value
  void setAttrValue(const String& name,const String& value);

  /*!
   * \brief Retourne l'attribut de nom \a name.
   *
   * Si l'attribut n'existe pas, si \a throw_exception vaut \a false retourne
   * un noeud nul, sinon lève une exception.
   */
  XmlNode attr(const String& name,bool throw_exception=false) const;

  /*!
   * \brief Retourne l'attribut de nom \a name.
   * Si aucun attribut avec ce nom n'existe, un attribut avec comme
   * valeur la chaîne nul est créé et retourné.
   */
  XmlNode forceAttr(const String& name);

  /*!
   * \brief Supprime l'attribut de nom \a name de ce noeud.
   * Si ce noeud n'est pas élément, rien n'est effectué.
   */
  void removeAttr(const String& name) const;

  /*!
   * \brief Retourne le noeud élément du document.
   * \pre type()==DOCUMENT_NODE
   */
  XmlNode documentElement() const;

  /*!
   * \brief Retourne l'élément propriétaire de cet attribut.
   * \pre type()==ATTRIBUTE_NODE
   */
  XmlNode ownerElement() const;

  //! Supprime tous les noeuds fils
  void clear();

  /*!
   * \brief Noeud fils de celui-ci de nom \a name
   *
   * Si plusieurs noeuds portant ce nom existent, retourne le premier.
   * Si le noeud n'est pas trouvé, retourne un noeud nul
   */
  XmlNode child(const String& name) const;

  /*!
   * \brief Noeud fils de celui-ci de nom \a name
   *
   * Si plusieurs noeuds portant ce nom existent, retourne le premier.
   * Si le noeud n'est pas trouvé, lève une exception.
   */
  XmlNode expectedChild(const String& name) const;

  //! Ensemble des noeuds fils de ce noeud ayant pour nom \a name
  XmlNodeList children(const String& name) const;

  //! Ensemble des noeuds fils de ce noeud
  XmlNodeList children() const;

  //! Parent de ce noeud (null si aucun)
  XmlNode parent() const { return XmlNode(m_rm,m_node.parentNode()); }

  /*! \brief Ajoute \a child_node comme fils de ce noeud.
   *
   * Le noeud est ajouté après tous les fils.
   */
  void append(const XmlNode& child_node) { m_node.appendChild(child_node.domNode()); }
  //! Supprime le noeud fils \a child_node
  void remove(const XmlNode& child_node);
  //! Remplace le noeud fils \a ref_node par le noeud \a new_node
  void replace(const XmlNode& new_node,XmlNode& ref_node);
  //! Supprime ce noeud du document
  void remove();
  //! Premier fils
  XmlNode front() const { return XmlNode(m_rm,m_node.firstChild()); }
  //! Dernier fils
  XmlNode last() const { return XmlNode(m_rm,m_node.lastChild()); }
  //! Noeud suivant (nextSibling())
  XmlNode next() const { return XmlNode(m_rm,m_node.nextSibling()); }
  //! Noeud précédent (previousSibling())
  XmlNode prev() const { return XmlNode(m_rm,m_node.previousSibling()); }
  //! Retourne le noeud suivant ce noeud ayant le nom \a name.
  XmlNode nextWithName(const String& name) const;
  //! Retourne le noeud précédent ce noeud ayant le nom \a name.
  XmlNode prevWithName(const String& name) const;
  //! Retourne le noeud suivant ce noeud ayant le même type.
  XmlNode nextSameType() const;
  //! Retourne le noeud précédent ce noeud ayant le même type.
  XmlNode prevSameType() const;
  void operator++() { m_node = m_node.nextSibling(); }
  void operator--() { m_node = m_node.previousSibling(); }

  //! Vrai si le noeud est nul
  bool null() const { return m_node._null(); }
  bool operator!() const { return null(); }

  //! \internal
  dom::Node domNode() const { return m_node; }
  //! \internal
  void assignDomNode(const dom::Node& node);

  /*!
   * \brief Insère un noeud.
   * Insère le noeud \a new_child après le noeud \a ref_node.
   * Si \a new_child est \c nul, ne fait rien.
   * Si \a ref_node est \c nul, \a new_child est ajouté à la fin (comme append()). Sinon,
   * \a ref_node doit être fils de ce noeud et \a new_child est inséré après
   * \a ref_node.
   * En cas de succès, retourne le noeud ajouté (\a new_child), sinon le noeud nul.
   */
  XmlNode insertAfter(const XmlNode& new_child,const XmlNode& ref_node);

  /*!
   * \brief Retourne le fils de ce noeud ayant pour nom \a elem_name et
   * un attribut de nom \a attr_name avec pour valeur \a attr_value.
   */
  XmlNode childWithAttr(const String& elem_name,const String& attr_name,
			const String& attr_value) const;
  /*!
   * \brief Retourne le fils de ce noeud ayant pour nom \a elem_name et
   * un attribut de nom \c "name" avec pour valeur \a attr_value.
   */
  XmlNode childWithNameAttr(const String& elem_name,
			    const String& attr_value) const;

  /*!
   * \brief Retourne un noeud à partir d'une expression XPath.
   * \param xpath_expr Expression XPath.
   */
  XmlNode xpathNode(const String& xpath_expr) const;

  /*!
   * \brief Créé un noeud d'un type donné.
   *
   * Si type() ne vaut pas DOCUMENT_NODE, utilise ownerDocument() comme
   * fabrique.
   *
   * \param type type du noeud.
   * \param nom du noeud.
   * \param valeur du noeud.
   * \return le noeud créé.
   * \pre type()==DOCUMENT_NODE
   */
  XmlNode createNode(eType type,const String& name,const String& value);

  /*!
   * \brief Créé un noeud d'un type donné.
   *
   * Si type() ne vaut pas DOCUMENT_NODE, utilise ownerDocument() comme
   * fabrique.
   *
   * \param type type du noeud.
   * \param nom ou valeur du noeud dans le cas ou le noeud n'a pas de nom.
   * \return le noeud créé.
   */
  XmlNode createNode(eType type,const String& name_or_value);

  /*!
   * \brief Créé un noeud texte.
   * \param value valeur du noeud texte.
   * \return le noeud créé.
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
 * \brief Elément d'un arbre DOM.
 */
class ARCANE_CORE_EXPORT XmlElement
: public XmlNode
{
 public:
  /*! \brief Créé un élément fils de \a parent.
   * L'élément créé a pour nom \a name et pour valeur \a value. 
   * Il est ajouté à la fin de la liste des fils de \a parent.
   */
  XmlElement(XmlNode& parent,const String& name,const String& value);
  /*! \brief Créé un élément fils de \a parent.
   * L'élément créé a pour nom \a name et pour valeur \a value.
   * Il est ajouté à la fin de la liste des fils de \a parent.
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

