// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IXmlDocumentHolder.h                                        (C) 2000-2018 */
/*                                                                           */
/* Interface d'un gestionnaire d'un document DOM.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IXMLDOCUMENTHOLDER_H
#define ARCANE_IXMLDOCUMENTHOLDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XmlNode;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Xml
 * \brief Gestionnaire d'un document DOM.
 *
 * Cette classe encapsule le noeud document d'un arbre DOM.
 * Le destructeur de cette classe libère l'arbre DOM.
 * L'utilisateur doit bien faire attention à ne plus utiliser
 * un noeud de cet arbre après sa libération.
 */
class ARCANE_CORE_EXPORT IXmlDocumentHolder
{
 public:

  //! Libère les ressources
  virtual ~IXmlDocumentHolder() {}

 public:

  //! Créé et retourne un document nul.
  static IXmlDocumentHolder* createNull();

  /*!
   * \brief Charge un document XML.
   *
   * Lit et analyse le document XML de nom \a name dont les données sont dans \a buffer.
   *
   * L'instance retournée n'est jamais nulle.
   * L'appelant est propriétaire de l'instance retournée et doit
   * la détruire par l'opérateur delete.
   */
  static IXmlDocumentHolder*
  loadFromBuffer(Span<const Byte> buffer,const String& name,ITraceMng* tm);

  /*!
   * \brief Charge un document XML.
   *
   * Lit et analyse le document XML de nom \a name dont les données sont dans \a buffer.
   *
   * L'instance retournée n'est jamais nulle.
   * L'appelant est propriétaire de l'instance retournée et doit
   * la détruire par l'opérateur delete.
   */
  static IXmlDocumentHolder*
  loadFromBuffer(ByteConstSpan buffer,const String& name,ITraceMng* tm);

  /*!
   * \brief Charge un document XML.
   *
   * Lit et analyse le document XML contenu dans le fichier \a filename.
   *
   * L'instance retournée n'est jamais nulle.
   * L'appelant est propriétaire de l'instance retournée et doit
   * la détruire par l'opérateur delete.
   */
  static IXmlDocumentHolder*
  loadFromFile(const String& filename,ITraceMng* tm);

  /*!
   * \brief Charge un document XML.
   *
   * Lit et analyse le document XML contenu dans le fichier \a filename.
   *
   * L'instance retournée n'est jamais nulle.
   * L'appelant est propriétaire de l'instance retournée et doit
   * la détruire par l'opérateur delete.
   *
   * Si \a schema_filename est non nul, il indique le fichier XML contenant le schéma
   * utilisé pour valider le fichier XML.
   */
  static IXmlDocumentHolder*
  loadFromFile(const String& filename,const String& schema_filename,ITraceMng* tm);

 public:

  //! Noeud document. \c Ce noeud est nul si le document n'existe pas.
  virtual XmlNode documentNode() =0;

  //! Clone ce document
  virtual IXmlDocumentHolder* clone() =0;

  //! Sauvegarde ce document dans le tableau \a bytes.
  virtual void save(ByteArray& bytes) =0;

  //! Sauvegarde ce document et retourne la chaîne de caractères.
  virtual String save() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

