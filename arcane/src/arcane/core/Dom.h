// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Dom.h                                                       (C) 2000-2024 */
/*                                                                           */
/* Implémentation d'un DOM1+DOM2+DOM3(core).                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DOM_H
#define ARCANE_DOM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/DomDeclaration.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IStringImpl;
class IXmlDocumentHolder;
}

namespace Arcane::dom
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern NodePrv* toNodePrv(const Node& node);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef String DOMString;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! @name  ExceptionCode
  An integer indicating the type of error generated.
  \note Other numeric codes are reserved for W3C for possible future
  use.
  \relate DOMException
*/
//@{
/*! If index or size is negative, or greater than the allowed value */
const UShort INDEX_SIZE_ERR = 1;
/*! If the specified range of text does not fit into a DOMString */
const UShort DOMSTRING_SIZE_ERR = 2;
//! If any node is inserted somewhere it doesn't belong 
const UShort HIERARCHY_REQUEST_ERR = 3;
/*! If a node is used in a different document than the one that
  created it (that doesn't support it) */
const UShort WRONG_DOCUMENT_ERR = 4;
/*! If an invalid or illegal character is specified, such as in a
  name. See production 2 in the XML specification for the definition of
  a legal character, and production 5 for the definition of a legal name
  character. */
const UShort INVALID_CHARACTER_ERR = 5;
//! If data is specified for a node which does not support data 
const UShort NO_DATA_ALLOWED_ERR = 6;
/*! If an attempt is made to modify an object where modifications are
  not allowed */
const UShort NO_MODIFICATION_ALLOWED_ERR = 7;
/*! If an attempt is made to reference a node in a context where it
  does not exist */
const UShort NOT_FOUND_ERR = 8;
/*! If the implementation does not support the requested type of
  object or operation */
const UShort NOT_SUPPORTED_ERR = 9;
/*! If an attempt is made to add an attribute that is already in use
  elsewhere */
const UShort INUSE_ATTRIBUTE_ERR = 10;
/*! If an attempt is made to use an object that is not, or is no
  longer, usable */
const UShort INVALID_STATE_ERR = 11;
/*! If an invalid or illegal string is specified */
const UShort SYNTAX_ERR = 12;
/*! If an attempt is made to modify the type of the underlying object */
const UShort INVALID_MODIFICATION_ERR = 13;
/*! If an attempt is made to create or change an object in a way which
  is incorrect with regard to namespaces */
const UShort NAMESPACE_ERR = 14;
/*! If a parameter or an operation is not supported by the underlying
  object */
const UShort INVALID_ACCESS_ERR = 15;
//@}

const UShort NOT_IMPLEMENTED_ERR = 2500;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT DOMException
{
 public:
  DOMException(UShort _code) : code(_code) {}
 public:
  UShort code; //!< The code of the exception
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT DOMWriter
{
 public:
  /*! @name Constructors and Destructors */
  //@{
  DOMWriter();
  DOMWriter(DOMWriterPrv*);
  DOMWriter(const DOMWriter& dw); 
 ~DOMWriter();
  //@}
  const DOMWriter& operator=(const DOMWriter& from);
 public:
  ByteUniqueArray writeNode(const Node& node) const;
  void encoding(const String& encoding);
  String encoding() const;
 private:
  DOMWriterPrv* m_p;
  DOMWriterPrv* _impl() const;
  bool _null() const;
  void _checkValid() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT DOMImplementation
{
 public:

  /*! @name DOM Level 1 operations */
  //@{
  bool hasFeature(const DOMString& feature,const DOMString& version) const;
  //@}

  /*! @name DOM Level 2 operations */
  //@{
  DocumentType createDocumentType(const DOMString& qualified_name,const DOMString& public_id,
                                  const DOMString& system_id) const;

 private:
  Document createDocument(const DOMString& namespace_uri,const DOMString& qualified_name,
                          const DocumentType& doctype) const;
 public:
  //@}

  /*! @name DOM Level 3 operations */
  //@{
  DOMImplementation getInterface(const DOMString& feature) const;
  //@}

  DOMWriter createDOMWriter() const;

 public:
  /*! @name Constructors and Destructors */
  //@{
  DOMImplementation();
  DOMImplementation(ImplementationPrv*);
  ~DOMImplementation();
  //@}

 public:
  //! Les méthodes suivantes sont internes à Arcane.
  //@{
  IXmlDocumentHolder* _newDocument();
  IXmlDocumentHolder* _load(const String& fname,ITraceMng* msg,const String& schemaname);
  IXmlDocumentHolder* _load(const String& fname,ITraceMng* msg,const String& schemaname, ByteConstArrayView schema_data);
  IXmlDocumentHolder* _load(ByteConstSpan buffer,const String& name,ITraceMng* trace);
  void _save(ByteArray& bytes,const Document& document,int indent_level);
  String _implementationName() const;
  //@}
 public:
  static void initialize();
  static void terminate();
 private:
  ImplementationPrv* m_p;
  ImplementationPrv* _impl() const;
  void _checkValid() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_CORE_EXPORT bool operator==(const Node& n1,const Node& n2);
ARCANE_CORE_EXPORT bool operator!=(const Node& n1,const Node& n2);
//class NodePrv;

/*!
 * \internal
 * \ingroup Xml
 */
class ARCANE_CORE_EXPORT Node
{
 public:

  /*! @name NodeType
    An integer indicating which type of node this is.
    \note Numeric codes up to 200 are reserved to W3C for possible future use.
   */
  //@{
  //! The node is an Element
  static const UShort ELEMENT_NODE = 1;
  //! The node is an Attr
  static const UShort ATTRIBUTE_NODE = 2;
  //! The node is a Text node
  static const UShort TEXT_NODE = 3;
  //! The node is a CDATASection
  static const UShort CDATA_SECTION_NODE = 4;
  //! The node is an EntityReference
  static const UShort ENTITY_REFERENCE_NODE = 5;
  //! The node is an Entity
  static const UShort ENTITY_NODE = 6;
  //! The node is a ProcessingInstruction
  static const UShort PROCESSING_INSTRUCTION_NODE = 7;
  //! The node is a Comment
  static const UShort COMMENT_NODE = 8;
  //! The node is a Document
  static const UShort DOCUMENT_NODE = 9;
  //! The node is a DocumentType
  static const UShort DOCUMENT_TYPE_NODE = 10;
  //! The node is a DocumentFragment
  static const UShort DOCUMENT_FRAGMENT_NODE = 11;
  //! The node is a Notation
  static const UShort NOTATION_NODE = 12;
  //@}

 public:

  /*! @name Attribute nodeName (DOM Level 1) */
  //@{
  DOMString nodeName() const;
  //@}

  /*! @name Attribute nodeValue (DOM Level 1) */
  //@{
  DOMString nodeValue() const;
  void nodeValue(const DOMString& value) const;
  //@}

  /*! @name Attribute nodeType (DOM Level 1) */
  //@{
  UShort nodeType() const;
  //@}

  /*! @name Attribute parentNode (DOM Level 1) */
  //@{
  Node parentNode() const;
  //@}

  /*! @name Attribute childNodes (DOM Level 1) */
  //@{
  NodeList childNodes() const;
  //@}

  /*! @name Attribute firstChild() (DOM Level 1) */
  //@{
  Node firstChild() const;
  //@}

  /*! @name Attribute lastChild() (DOM Level 1) */
  //@{
  Node lastChild() const;
  //@}

  /*! @name Attribute previousSibling() (DOM Level 1) */
  //@{
  Node previousSibling() const;
  //@}

  /*! @name Attribute nextSibling() (DOM Level 1) */
  //@{
  Node nextSibling() const;
  //@}

  /*! @name Attribute attributes() (DOM Level 1) */
  //@{
  NamedNodeMap attributes() const;
  //@}

  /*! @name Attribute ownerDocument() (DOM Level 2) */
  //@{
  Document ownerDocument() const;
  //@}

  /*! @name DOM Level 1 operations */
  //@{
  Node insertBefore(const Node& new_child,const Node& ref_child) const;
  Node replaceChild(const Node& new_child,const Node& old_child) const;
  Node removeChild(const Node& old_child) const;
  Node appendChild(const Node& new_child) const;
  bool hasChildNodes() const;
  Node cloneNode(bool deep) const;
  //@}

  /*! @name Attribute prefix() (DOM Level 2). */
  //@{
  DOMString prefix() const;
  void prefix(const DOMString& new_prefix) const;
  //@}

  /*! @name DOM Level 2 operations */
  //@{
  void normalize() const;
  bool isSupported(const DOMString& feature,const DOMString& version) const;
  DOMString namespaceURI() const;

  DOMString localName() const;
  //@}

  /*! @name TreePosition
    A bitmask indicating the relative tree position of a node with
    respect to another node.

    Issue TreePosition-1: 
    Should we use fewer bits? 
    
    Issue TreePosition-2: 
    How does a node compare to itself? 
  */
  //@{
  //! The node precedes the reference node
  static const UShort TREE_POSITION_PRECEDING = 0x01;
  //! The node follows the reference node
  static const UShort TREE_POSITION_FOLLOWING = 0x02;
  //! The node is an ancestor of the reference node
  static const UShort TREE_POSITION_ANCESTOR = 0x04;
  //! The node is a descendant of the reference node
  static const UShort TREE_POSITION_DESCENDANT = 0x08;
  /*! The two nodes have the same position. This is the case of two
    attributes that have the same ownerElement() */
  static const UShort TREE_POSITION_SAME = 0x10;
  /*! The two nodes have the exact same position. This is never the
    case of two attributes, even when they have the same
    ownerElement. Two nodes that have the exact same position have the
    same position, though the reverse may not be true.
  */
  static const UShort TREE_POSITION_EXACT_SAME = 0x20;
  //! The two nodes are disconnected, they do not have any common ancestor
  static const UShort TREE_POSITION_DISCONNECTED = 0x00;
  //@}

  /*! @name Attribute textContent() (DOM Level 3) */
  //@{
  DOMString textContent() const;
  void textContent(const DOMString& value) const;
  //@}

  /*! @name Attribute baseURI() (DOM Level 3) */
  //@{
  DOMString baseURI() const;
  //@}
  

  /*! @name DOM Level 3 operations */
  //@{
  bool isSameNode(const Node& node) const;
  UShort compareTreePosition(const Node& other) const;
  bool isEqualNode(const Node& other) const;
  Node getInterface(const DOMString& feature) const;
  DOMString lookupNamespacePrefix(const DOMString& namespace_uri,bool use_default) const;
  bool isDefaultNamespace(const DOMString& namespace_uri) const;
  DOMString lookupNamespaceURI(const DOMString& prefix) const;
  //void normalizeNS() const;
  DOMObject setUserData(const DOMString& key,const DOMObject& data,
                        const UserDataHandler& handler) const;
  DOMObject getUserData(const DOMString& key) const;
  //@}

  /*!
   * \brief Détruit le noeud.
   *
   * Le noeud ne doit pas appartenir à un document.
   *
   * Le noeud ne doit plus être utilisé par la suite.
   *
   * Cette méthode ne fait pas partie du DOM mais est nécessaire pour
   * certaines implémentation pour supprimer la mémoire associée à un noeud.
   */
  void releaseNode();

 public:
  
  bool _null() const;

 public:

  friend class IDOM_Node;
  friend class IDOM_Document;
  friend class Attr;
  friend class Element;
  friend class Document;
  friend class DOMImplementation;
  friend class NamedNodeMap;
	friend class CharacterData;
	friend class Text;
  friend class DOMWriter;
  friend bool ARCANE_CORE_EXPORT operator==(const Node& n1,const Node& n2);

 protected:

  NodePrv* m_p; //!< Implémentation de la classe.
  //void* m_p;
  void _assign(const Node&);

 public:
  Node();
  Node(const Node&);
  virtual ~Node();
  const Node& operator=(const Node& from);
 public:
  Node(NodePrv*);
 protected:
  void _checkValid() const;
 protected:
  NodePrv* _impl() const;
  friend NodePrv* toNodePrv(const Node& node);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \ingroup Xml
 */
class ARCANE_CORE_EXPORT Document
: public Node
{
 public:

  /*! @name Attribute doctype() (DOM Level 1) */
  //@{
  DocumentType doctype() const;
  //@}

  /*! @name Attribute implementation() (DOM Level 1) */
  //@{
  DOMImplementation implementation() const;
  //@}

  /*! @name Attribute documentElement() (DOM Level 1) */
  //@{
  Element documentElement() const;
  //@}

  //! @name DOM Level 1 operations
  //@{
  Element createElement(const DOMString& tag_name) const;
  DocumentFragment createDocumentFragment() const;
  Text createTextNode(const DOMString& data) const;
  Comment createComment(const DOMString& data) const;
  CDATASection createCDATASection(const DOMString& data) const;
  ProcessingInstruction createProcessingInstruction(const DOMString& target,
                                                    const DOMString& data) const;
  Attr createAttribute(const DOMString& name) const;
  EntityReference createEntityReference(const DOMString& name) const;
  NodeList getElementsByTagName(const DOMString& tagname) const;
  //@}

  //! @name DOM Level 2 operations
  //@{
  Node importNode(const Node& imported_node,bool deep) const;
  Element createElementNS(const DOMString& namespace_uri,const DOMString& qualified_name) const;
  Attr createAttributeNS(const DOMString& namespace_uri,const DOMString& qualified_name) const;
  NodeList getElementsByTagNameNS(const DOMString& namespace_uri,const DOMString& local_name) const;
  Element getElementById(const DOMString& element_id) const;
  //@}

  /*! @name Attribute actualEncoding() (DOM Level 3) */
  //@{
  DOMString actualEncoding() const;
  void actualEncoding(const DOMString&) const;
  //@}

  /*! @name Attribute encoding() (DOM Level 3) */
  //@{
  DOMString encoding() const;
  void encoding(const DOMString&) const;
  //@}

  /*! @name Attribute standalone() (DOM Level 3) */
  //@{
  bool standalone() const;
  void standalone(bool) const;
  //@}

  /*! @name Attribute strictErrorChecking() (DOM Level 3) */
  //@{
  bool strictErrorChecking() const;
  void strictErrorChecking(bool) const;
  //@}

  /*! @name version() (DOM Level 3) */
  //@{
  DOMString version() const;
  void version(const DOMString&) const;
  //@}

  /*! @name Attribute errorHandler() (DOM Level 3) */
  //@{
  DOMErrorHandler errorHandler() const;
  void errorHandler(const DOMErrorHandler&) const;
  //@}

  /*! @name Attribute documentURI() (DOM Level 3) */
  //@{
  DOMString documentURI() const;
  void documentURI(const DOMString&) const;
  //@}

  //! @name DOM Level 3 operations
  //@{
  Node adoptNode(const Node& source) const;
  void normalizeDocument();
  /*
  void setNormalizationFeature(const DOMString& name,bool state);
  bool getNormalizationFeature(const DOMString& name) const;
  */
  Node renameNode(const Node& node,const DOMString& namespace_uri,
                  const DOMString& name);
  //@}

 public:
  friend class IDOM_Document;
  friend class DOMImplementation;
 public:
  Document();
  Document(const Node&);
  Document(DocumentPrv*);
 private:
  DocumentPrv* _impl() const;
};

class ARCANE_CORE_EXPORT DocumentFragment
: public Node
{
 public:
  DocumentFragment();
  DocumentFragment(DocumentFragmentPrv*);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \ingroup Xml
 */
class ARCANE_CORE_EXPORT NodeList
{
 public:
  //! @name DOM Level 1 operations
  //@{
  Node item(ULong index) const;
  ULong length() const;
  //@}
 public:
  NodeList();
  NodeList(NodeListPrv*);
 private:
  NodeListPrv* _impl() const;
  NodeListPrv* m_p;
  void _checkValid() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \ingroup Xml
 */
class ARCANE_CORE_EXPORT CharacterData
: public Node
{
 public:
  /*! @name Attribute data (DOM Level 1) */
  //@{
  DOMString data() const;
  void data(const DOMString&) const;
  //@}

  /*! @name Attribute length (DOM Level 1) */
  //@{
  ULong length() const;
  //@}

  //! @name DOM Level 1 operations
  //@{
  DOMString substringData(ULong offset,ULong count) const;
  void appendData(const DOMString& arg) const;
  void insertData(ULong offset,const DOMString& arg) const;
  void deleteData(ULong offset,ULong count) const;
  void replaceData(ULong offset,ULong count,const DOMString& arg) const;
  //@}
 protected:
  CharacterData();
  CharacterData(const Node& from);
  CharacterData(const CharacterData& from);
  CharacterData(CharacterDataPrv*);
  CharacterDataPrv* _impl() const;
};

class ARCANE_CORE_EXPORT Attr
: public Node
{
 public:
  /*! @name Attribute name (DOM Level 1) */
  //@{
  DOMString name() const;
  //@}

  /*! @name Attribute specified (DOM Level 1) */
  //@{
  bool specified() const;
  //@}

  /*! @name Attribute value (DOM Level 1) */
  //@{
  DOMString value() const;
  void value(const DOMString& str) const;
  //@}

  /*! @name Attribute ownerElement (DOM Level 2) */
  //@{
  Element ownerElement() const;
  //@}

 public:
  friend class IDOM_Attr;
  friend class IDOM_Node;
  friend class Element;
 public:
  Attr() = default;
  Attr(const Node&);
  Attr(AttrPrv*);
 private:
  AttrPrv* _impl() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT Element
: public Node
{
 public:

  /*! @name Attribut tagName (DOM Level 1) */
  //@{
  DOMString tagName() const;
  //@}

  /*! @name DOM Level 1 operations */
  //@{
  DOMString getAttribute(const DOMString& name) const;
  void setAttribute(const DOMString& name,const DOMString& value) const;
  void removeAttribute(const DOMString& name) const;
  Attr getAttributeNode(const DOMString& name) const;
  Attr setAttributeNode(const Attr& new_attr) const;
  Attr removeAttributeNode(const Attr& old_attr) const;
  NodeList getElementsByTagName(const DOMString& name) const;
  //@}
  //@{ @name DOM Level 2 operations
  DOMString getAttributeNS(const DOMString& namespace_uri,const DOMString& local_name) const;
  void setAttributeNS(const DOMString& namespace_uri,const DOMString& qualified_name,
		      const DOMString& value) const;
  void removeAttributeNS(const DOMString& namespace_uri,const DOMString& local_name) const;
  Attr getAttributeNodeNS(const DOMString& namespace_uri,const DOMString& local_name) const;
  Attr setAttributeNodeNS(const Attr& new_attr) const;
  NodeList getElementsByTagNameNS(const DOMString& namespace_uri,const DOMString& local_name) const;
  bool hasAttribute(const DOMString& name) const;
  bool hasAttributeNS(const DOMString& namespace_uri,const DOMString& local_name) const;
  //@}
 public:
  Element();
  Element(const Element&);
  Element(const Node&);
 private:
  Element(ElementPrv*);
  friend class IDOM_Element;
  friend class Attr;
  friend class Document;
  ElementPrv* _impl() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT Text
: public CharacterData
{
 public:
  /*! @name DOM Level 1 operations */
  //@{
  Text splitText(ULong offset) const;
  //@}
  /*! @name Attribute isWhiteSpaceInElementContent (DOM Level 3) */
  //@{
  bool isWhiteSpaceInElementContent() const;
  //@}
  /*! @name Attribute wholeText() (DOM Level 3) */
  //@{
  DOMString wholeText() const;
  //@}
  /*! @name DOM Level 3 operations */
  //@{
  Text replaceWholeText(const DOMString& content) const;
  //@}
 public:
  Text();
  Text(const Text&);
  Text(const Node&);
  Text(TextPrv*);
  TextPrv* _impl() const;
  friend class Document;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT Comment
: public CharacterData
{
 public:
  Comment();
  Comment(CommentPrv*);
 private:
  CommentPrv* _impl() const;
};

class ARCANE_CORE_EXPORT CDATASection
: public Text
{
 public:
  CDATASection();
  CDATASection(CDATASectionPrv*);
 private:
  CDATASectionPrv* _impl() const;
};

class DocumentType
: public Node
{
 public:
  /*! @name Attribute name() (DOM Level 1) */
  //@{
  DOMString name() const;
  //@}

  /*! @name Attribute entities() (DOM Level 1) */
  //@{
  NamedNodeMap entities() const;
  //@}

  /*! @name Attribute notations() (DOM Level 1) */
  //@{
  NamedNodeMap notations() const;
  //@}
  /*! @name Attribute publicId() (DOM Level 2) */
  //@{
  DOMString publicId() const;
  //@}

  /*! @name Attribute systemId() (DOM Level 2) */
  //@{
  DOMString systemId() const;
  //@}

  /*! @name Attribute internalSubset() (DOM Level 2) */
  //@{
  DOMString internalSubset() const;
  //@}
 public:
  DocumentType(DocumentTypePrv*);
  DocumentType();
 private:
  friend class Arcane::dom::DOMImplementation;
  DocumentTypePrv* _impl() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT Notation
: public Node
{
 public:
  /*! @name Attribute publicId() (DOM Level 1) */
  //@{
  DOMString publicId() const;
  //@}

  /*! @name Attribute systemId() (DOM Level 1) */
  //@{
  DOMString systemId() const;
  //@}
 private:
  NotationPrv* _impl() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT Entity
: public Node
{
 public:
  /*! @name Attribute publicId() (DOM Level 1) */
  //@{
  DOMString publicId() const;
  //@}

  /*! @name Attribute systemId() (DOM Level 1) */
  //@{
  DOMString systemId() const;
  //@}

  /*! @name Attribute notationName() (DOM Level 1) */
  //@{
  DOMString notationName() const;
  //@}
  /*! @name Attribute actualEncoding() (DOM Level 3) */
  //@{
  DOMString actualEncoding() const;
  void actualEncoding(const DOMString&) const;
  //@}

  /*! @name Attribute encoding() (DOM Level 3) */
  //@{
  DOMString encoding() const;
  void encoding(const DOMString&) const;
  //@}

  /*! @name Attribute version() (DOM Level 3) */
  //@{
  DOMString version() const;
  void version(const DOMString&) const;
  //@}
 private:
  EntityPrv* _impl() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class EntityReference
: public Node
{
 public:
  EntityReference();
  EntityReference(EntityReferencePrv*);
 private:
  EntityReferencePrv* _impl() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT ProcessingInstruction
: public Node
{
 public:

  /*! @name Attribute target() (DOM Level 1) */
  //@{
  DOMString target() const;
  //@}

  /*! @name Attribute data() (DOM Level 1) */
  //@{
  DOMString data() const;
  void data(const DOMString& value) const;
  //@}
 public:
  ProcessingInstruction();
  ProcessingInstruction(ProcessingInstructionPrv*);
 private:
  ProcessingInstructionPrv* _impl() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT NamedNodeMap
{
 public:
  /*! @name Attribute length (DOM Level 1)*/
  //@{
  ULong length() const;
  //@}
  /*! @name DOM Level 1 operations */
  //@{
  Node getNamedItem(const DOMString& name) const;
  Node setNamedItem(const Node& arg) const;
  Node removeNamedItem(const DOMString& name) const;
  Node item(ULong index) const;
  //@}
  /*! @name DOM Level 2 operations */
  //@{
  Node getNamedItemNS(const DOMString& namespace_uri,const DOMString& local_name) const;
  Node setNamedItemNS(const Node& arg) const;
  Node removeNamedItemNS(const DOMString& namespace_uri,const DOMString& local_name) const;
  //@}
 public:
  bool _null() const;
 public:
  friend class IDOM_Node;
  friend class IDOM_Element;
  NamedNodeMap();
  NamedNodeMap(NamedNodeMapPrv*);
  ~NamedNodeMap();
  NamedNodeMap(const NamedNodeMap& from);
  const NamedNodeMap& operator=(const NamedNodeMap& from);
 private:
  //AutoRefT<NamedNodeMapPrv> m_p;
  NamedNodeMapPrv* m_p;
  NamedNodeMapPrv* _impl() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DOMImplementationSource
{
 public:
  DOMImplementation getDOMImplementation(const DOMString& features) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UserDataHandler
{
 public:
  //! @name OperationType
  //@{
  //! The node is cloned.
  static const UShort  CLONED     = 1;
  //! The node is imported.
  static const UShort  IMPORTED   = 2;
  //! The node is deleted.
  static const UShort  DELETED    = 3;
  //@}

  void handle(UShort operation,const DOMString& key,const DOMObject& data,
	      const Node& src,const Node& dest) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DOMError
{
 public:
  //! @name ErrorType
  //@{
  //! The severity of the error described by the DOMError is warning 
  static const UShort SEVERITY_WARNING       = 0;
  //! The severity of the error described by the DOMError is error
  static const UShort SEVERITY_ERROR         = 1;
  //! The severity of the error described by the DOMError is fatal error
  static const UShort SEVERITY_FATAL_ERROR   = 2;
  //@}

  /*! @name Attribute severity (DOM Level 3) */
  //@{
  UShort severity() const;
  //@}

  /*! @name Attribute message (DOM Level 3) */
  //@{
  DOMString message() const;
  //@}

  /*! @name Attribute exception (DOM Level 3) */
  //@{
  DOMObject relatedException() const;
  //@}

  /*! @name Attribute location (DOM Level 3) */
  //@{
  DOMLocator location() const;
  //@}
 public:
  DOMError();
  DOMError(DOMErrorPrv*);
  ~DOMError();
  DOMError(const DOMError& from);
  const DOMError& operator=(const DOMError& from);
 private:
  DOMErrorPrv* m_p;
  DOMErrorPrv* _impl() const;
  bool _null() const;
  void _checkValid() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DOMErrorHandler
{
 public:
  /*! @name DOM Level 3 operations */
  //@{
  bool handleError(const DOMError& error) const;
  //@}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DOMLocator
{
 public:
  /*! @name Attribute lineNumber (DOM Level 3) */
  //@{
  long lineNumber() const;
  //@}
  /*! @name Attribute columnNumber (DOM Level 3) */
  //@{
  long columnNumber() const;
  //@}
  /*! @name Attribute offset (DOM Level 3) */
  //@{
  long offset() const;
  //@}
  /*! @name Attribute errorNode (DOM Level 3) */
  //@{
  Node errorNode() const;
  //@}
  /*! @name Attribute uri (DOM Level 3) */
  //@{
  DOMString uri() const;
  //@}
 public:
  friend class DOMError;
  DOMLocator();
  DOMLocator(DOMLocatorPrv*);
  ~DOMLocator();
  DOMLocator(const DOMLocator& from);
  const DOMLocator& operator=(const DOMLocator& from);
 private:
  DOMLocatorPrv* m_p;
  DOMLocatorPrv* _impl() const;
  bool _null() const;
  void _checkValid() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! @name XPathExceptionCode
//@{
/*!
  If the expression is not a legal expression according to the rules
  of the specific XPathEvaluator or contains namespace prefixes which
  are not in scope according to the specified XPathNSResolver. If the
  XPathEvaluator was obtained by casting the document, the expression
  must be XPath 1.0 with no special extension functions.
*/
const unsigned short INVALID_EXPRESSION_ERR = 1;
/*!
  If the expression cannot be converted to return the specified type. 
  Interface XPathEvaluator
*/
const unsigned short TYPE_ERR = 2;
//@}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XPathException
{
 public:
  unsigned short code;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XPathEvaluator
{
 public:
  XPathExpression createExpression(const DOMString& expression, 
				   const XPathNSResolver& resolver) const;
  XPathResult createResult() const;
  XPathNSResolver createNSResolver(const Node& node_resolver) const;
  XPathResult evaluate(const DOMString& expression, 
		       const Node& context_node, 
		       const XPathNSResolver& resolver, 
		       UShort type, 
		       const XPathResult& result) const;
  
  XPathResult evaluateExpression(const XPathExpression& expression, 
				 const Node& context_node, 
				 UShort type, 
				 const XPathResult& result) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XPathExpression
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XPathNSResolver
{
 public:
  DOMString lookupNamespaceURI(const DOMString& prefix) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XPathResult
{
 public:
  //! @name XPathResultType
  //@{
  /*!
    This code does not represent a specific type. An evaluation of an
    XPath expression will never produce this type. If this type is
    requested, then the evaluation must return whatever type Integerly
    results from evaluation of the expression.
   */
  static const UShort ANY_TYPE = 0;
  //! The result is a number as defined by XPath 1.0.
  static const UShort NUMBER_TYPE = 1;
  //! The result is a string as defined by XPath 1.0.
  static const UShort STRING_TYPE = 2;
  //! The result is a boolean as defined by XPath 1.0. 
  static const UShort BOOLEAN_TYPE = 3;
  //! The result is a node set as defined by XPath 1.0.
  static const UShort NODE_SET_TYPE = 4;
  /*!
    The result is a single node, which may be any node of the node set
    defined by XPath 1.0, or null if the node set is empty. This is a
    convenience that permits optimization where the caller knows that
    no more than one such node exists because evaluation can stop
    after finding the one node of an expression that would otherwise
    return a node set (of type NODE_SET_TYPE).

    Where it is possible that multiple nodes may exist and the first
    node in document order is required, a NODE_SET_TYPE should be
    processed using an ordered iterator, because there is no order
    guarantee for a single node.
   */
  static const UShort SINGLE_NODE_TYPE = 5;
  //@}

  /*! @name Attribute resultType (DOM Level 3) */
  //@{
  UShort resultType() const;
  //@}

  /*! @name Attribute numberValue (DOM Level 3) */
  //@{
  double numberValue() const;
  //@}

  /*! @name AttributestringValue (DOM Level 3) */
  //@{
  DOMString stringValue() const;
  //@}

  /*! @name Attribute booleanValue (DOM Level 3) */
  //@{
  bool booleanValue() const;
  //@}

  /*! @name Attribute singleNodeValue (DOM Level 3) */
  //@{
  Node singleNodeValue() const;
  //@}

  /*! @name DOM Level 3 operations */
  //@{
  XPathSetIterator getSetIterator(bool ordered) const;
  XPathSetSnapshot getSetSnapshot(bool ordered) const;
  //@}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XPathSetIterator
{
 public:

  Node nextNode() const;
};

class XPathSetSnapshot
{
 public:
  /*! @name Attribute length (DOM Level 3) */
  //@{
  ULong length() const;
  //@}

  /*! @name DOM Level 3 operations */
  //@{
  Node item(ULong index) const;
  //@}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XPathNamespace
: public Node
{
 public:
  //! @name XPathNodeType
  //@{
  //! The node is a Namespace. 
  static const UShort XPATH_NAMESPACE_NODE = 13;
  //@}

  /*! @name Attribute ownerElement (DOM Level 3) */
  //@{
  Element ownerElement() const;
  //@}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

