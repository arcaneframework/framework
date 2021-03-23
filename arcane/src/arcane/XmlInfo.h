// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* XmlInfo.h                                                   (C) 2000-2006 */
/*                                                                           */
/* Informations sur un fichier XML.                                          */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_XMLINFO_H
#define ARCANE_XMLINFO_H

#include "arcane/utils/Iostream.h"
#include "arcane/ArcaneException.h"
#include "arcane/IApplication.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/IIOMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Informations sur un fichier XML.
 *
 * Classe de base des classes stockant les informations lues dans les
 * fichiers XML. Un élément XML nommé XXX aura ses informations stockées
 * dans une classe XXXInfo.
 */
class XmlInfo
{
 public:
  XmlInfo(IApplication* mng=0) : m_mng(mng) {}
  virtual ~XmlInfo() {};
  
 public:
  /**
   * Leve une exception concernant l'élément XML "element" avec le message "msg".
   * @param element élément XML concerné par l'exception.
   * @param msg message de l'exception.
   */
  static void error(const XmlNode& element, const String msg)
  {
    OCStringStream msgStr;
    msgStr() << "** Error: Element <" << element.name()
             << ">: " << msg << '.';
    throw InternalErrorException("XmlInfo::_error", msgStr.str());
  }

  /**
   * Affiche le message d'avertissement "msg" concernant l'élément XML "element".
   * @param element élément XML concerné par l'exception.
   * @param msg message de l'exception.
   */
  static void warning(const XmlNode& element, const String msg)
  {
    cerr << "** Warning: Element <" << element.name()
         << ">: " << msg << ".\n";
  }

  /**
   * Leve une exception concernant l'élément XML "element" en précisant que
   * l'attribut "attr_name" a été oublié dans le fichier XML.
   * @param element élément XML concerné par l'exception.
   * @param attr_name nom de l'attribut concerné.
   */
  static void attrError(const XmlNode& element, const char* attr_name)
  {
    OCStringStream msg;
    msg() << "Attribute \"" << attr_name << "\" not specified";
    error(element, String(msg.str()));
  }

  /**
   * Retourne le noeud racine du fichier XML nommé "file_name". Si un nom
   * de schéma est passé en argument, la conformité du fichier au schéma
   * est vérifié.
   * @return le noeud XML racine du document.
   * @param file_name le nom du fichier XML.
   * @param mng le gestionnaire d'entrées/sorties Arcane.
   * @param schema_name le nom du schéma associé au fichier (optionnel).
   */
  static XmlNode rootNode(IIOMng* mng,
                          const String& file_name,
                          const String& schema_name = String())
  {
    IXmlDocumentHolder* xml_doc = mng->parseXmlFile(file_name.local(),
                                                    schema_name.local());
    if (!xml_doc)
    {
      OCStringStream s;
      s() << "Can't read the file \"" << file_name << "\"";
      throw InternalErrorException("XmlInfo::rootNode",s.str());
    }
    return xml_doc->documentNode().documentElement();
  }

  protected:
    /** Gestionnaire de l'application. */
   IApplication* m_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
